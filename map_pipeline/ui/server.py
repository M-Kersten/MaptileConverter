#!/usr/bin/env python3
"""A small local web UI for the pipeline.

    python ui/server.py

Then open http://127.0.0.1:8765. Pick an area on the map, press Build, watch the
log, and collect the files.

Standard library only. The pipeline itself is run as a subprocess exactly the
way you would run it by hand, so the UI cannot drift away from the CLI: it
writes a config file and calls ``pipeline.py --config``.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = REPO_ROOT / "ui"
sys.path.insert(0, str(REPO_ROOT))

GEOCODER = "https://api.pdok.nl/bzk/locatieserver/search/v3_1/free"
TILE_BASE = (
    "https://service.pdok.nl/hwh/luchtfotorgb/wmts/v1_0/"
    "Actueel_orthoHR/EPSG:28992"
)

# Keep finished jobs around so a page reload can still show the result.
MAX_JOBS = 24

# Small in-memory tile cache for the fallback route, so panning back over
# somewhere you have already been costs nothing.
TILE_CACHE: dict[str, bytes] = {}
TILE_CACHE_MAX = 400
TILE_CACHE_LOCK = threading.Lock()


# Cost model for the run-time estimate. Each stage costs a constant, plus a
# rate per square kilometre, plus a rate per aerial megapixel. The numbers come
# from measured runs over Utrecht: a 1 km area at 4096 px takes about 220 s
# without previews and about 470 s with them.
#
# These are only the starting point. `speed_factor` rescales the whole model
# from the timings of previous runs, so the estimate adapts to the machine it is
# actually on rather than to the one it was calibrated on.
STAGE_COSTS: dict[str, dict[str, float]] = {
    #                      constant  per km2  per megapixel
    "terrain":            {"c": 1.0, "a": 8.0,  "m": 0.0},
    "aerial":             {"c": 1.0, "a": 0.0,  "m": 0.75},
    "surfaces":           {"c": 1.0, "a": 30.0, "m": 2.5},
    "usage":              {"c": 0.5, "a": 5.0,  "m": 0.0},
    "buildings":          {"c": 2.0, "a": 75.0, "m": 0.0},
    "trees":              {"c": 1.0, "a": 23.0, "m": 0.0},
    "furniture":          {"c": 0.5, "a": 3.0,  "m": 0.0},
    "facade":             {"c": 1.0, "a": 0.0,  "m": 0.0},
    "scene":              {"c": 0.3, "a": 0.0,  "m": 0.0},
    "blender":            {"c": 2.0, "a": 8.0,  "m": 0.0},
    # Rendering three views costs more than every data stage together, and
    # varies with what is in the scene rather than with area alone, so this is
    # the loosest term in the model.
    "preview":            {"c": 5.0, "a": 160.0, "m": 0.0},
    "validation":         {"c": 3.0, "a": 2.0,  "m": 0.3},
}

# Fixed overhead per run: interpreter start, capability documents, TLS setup.
BASE_OVERHEAD_S = 12.0


def estimate_seconds(payload: dict, speed: float = 1.0) -> dict:
    """Predict how long a run with these settings will take."""
    bbox = payload.get("bbox") or {}
    try:
        width = abs(float(bbox["xmax"]) - float(bbox["xmin"]))
        height = abs(float(bbox["ymax"]) - float(bbox["ymin"]))
    except (KeyError, TypeError, ValueError):
        width = height = 1000.0
    area_km2 = max(width * height / 1e6, 1e-6)
    megapixels = int(payload.get("size_px", 4096)) ** 2 / 1e6

    # Either a list of selected source ids, or one flag per source.
    if "sources" in payload:
        chosen = set(payload["sources"] or [])

        def on(source_id: str) -> bool:
            return source_id in chosen
    else:

        def on(source_id: str) -> bool:
            return bool(payload.get(source_id, True))

    enabled = {
        "terrain": True,
        "aerial": True,
        "buildings": True,
        "surfaces": on("water") or on("land_cover"),
        "usage": on("usage"),
        "trees": on("trees"),
        "furniture": on("furniture"),
        "facade": True,
        "scene": True,
        "blender": True,
        "preview": bool(payload.get("preview", True)),
        "validation": True,
    }

    breakdown: dict[str, float] = {}
    for stage, cost in STAGE_COSTS.items():
        if not enabled.get(stage):
            continue
        seconds = cost["c"] + cost["a"] * area_km2 + cost["m"] * megapixels
        breakdown[stage] = round(seconds * speed, 1)

    total = BASE_OVERHEAD_S * speed + sum(breakdown.values())
    return {
        "seconds": round(total, 1),
        "breakdown": breakdown,
        "area_km2": round(area_km2, 4),
        "megapixels": round(megapixels, 2),
        "speed_factor": round(speed, 3),
    }


def measured_speed_factor() -> tuple[float, int]:
    """Scale the model by how past runs on this machine actually went.

    One factor over the whole model rather than refitting every coefficient:
    with a handful of runs that is all the data supports, and it captures what
    actually differs between machines, which is how fast they are.
    """
    root = REPO_ROOT / "work"
    if not root.is_dir():
        return 1.0, 0

    ratios: list[float] = []
    for path in sorted(root.glob("*/timings.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            drivers = record["drivers"]
            actual = float(record["total_seconds"])
        except (OSError, ValueError, KeyError):
            continue
        if actual <= 0 or any(stage.get("failed") for stage in record.get("stages", [])):
            continue
        if drivers.get("skip_blender"):
            continue  # not a comparable run
        # Older records predate the flag; a run with no stages did no work,
        # which is what a preflight abort looks like.
        if not record.get("completed", bool(record.get("stages"))):
            continue

        # Rebuild the settings the model takes, from what the run recorded.
        side = (drivers["area_km2"] * 1e6) ** 0.5
        predicted = estimate_seconds(
            {
                "bbox": {"xmin": 0, "ymin": 0, "xmax": side, "ymax": side},
                "size_px": int(
                    drivers.get("aerial_size_px")
                    or round(drivers["aerial_megapixels"] ** 0.5 * 1000)
                ),
                "trees": drivers.get("trees", True),
                "water": drivers.get("surfaces", True),
                "land_cover": drivers.get("surfaces", True),
                "furniture": drivers.get("furniture", True),
                "usage": drivers.get("usage", True),
                "preview": drivers.get("preview", False),
            }
        )["seconds"]
        if predicted > 0:
            ratios.append(actual / predicted)

    if not ratios:
        return 1.0, 0
    ratios.sort()
    median = ratios[len(ratios) // 2]
    # Bounded so one odd run cannot make every estimate nonsense.
    return max(0.25, min(4.0, median)), len(ratios)


class Job:
    """One pipeline run."""

    def __init__(self, job_id: str, name: str, config: dict, estimate: float) -> None:
        self.id = job_id
        self.name = name
        self.config = config
        self.estimate = estimate
        self.lines: list[str] = []
        self.status = "starting"  # starting | running | done | failed | cancelled
        self.returncode: int | None = None
        self.started = time.time()
        self.finished: float | None = None
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.step = 0
        self.total_steps = 0
        self.stage = "starting"

    def log(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            self._track_progress(line)

    def _track_progress(self, line: str) -> None:
        """Follow the run by the step headers it already prints.

        The pipeline announces every stage as "Step N/M  name", so progress
        needs no extra protocol between the two.
        """
        match = re.search(r"Step (\d+)/(\d+)\s+(.+?)\s*$", line)
        if match:
            self.step = int(match.group(1))
            self.total_steps = int(match.group(2))
            self.stage = match.group(3)
            return
        # Previews run after the last numbered stage, and take longer than all
        # of them together, so they get their own label rather than looking
        # like a stall on validation.
        if "preview.py via" in line:
            self.stage = "rendering previews"
            self.step = max(self.step, self.total_steps)

    def snapshot(self, offset: int) -> dict:
        with self.lock:
            lines = self.lines[offset:]
            total = len(self.lines)
            step, total_steps, stage = self.step, self.total_steps, self.stage

        elapsed = (self.finished or time.time()) - self.started
        if self.status in ("done", "failed", "cancelled"):
            fraction = 1.0
        elif total_steps:
            fraction = min(0.99, step / (total_steps + 1))
        else:
            fraction = 0.0

        remaining = None
        if self.status in ("running", "starting") and self.estimate > 0:
            remaining = max(0.0, self.estimate - elapsed)

        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "returncode": self.returncode,
            "lines": lines,
            "offset": total,
            "elapsed": round(elapsed, 1),
            "step": step,
            "total_steps": total_steps,
            "stage": stage,
            "fraction": round(fraction, 4),
            "estimate": round(self.estimate, 1),
            "remaining": None if remaining is None else round(remaining, 1),
        }


JOBS: dict[str, Job] = {}
JOBS_ORDER: list[str] = []
JOBS_LOCK = threading.Lock()


def register(job: Job) -> None:
    with JOBS_LOCK:
        JOBS[job.id] = job
        JOBS_ORDER.append(job.id)
        while len(JOBS_ORDER) > MAX_JOBS:
            old = JOBS_ORDER.pop(0)
            JOBS.pop(old, None)


def build_config(payload: dict) -> dict:
    """Turn the form payload into a pipeline config."""
    bbox = payload["bbox"]
    config = {
        "name": payload["name"],
        "bbox": {
            "crs": "EPSG:28992",
            "xmin": float(bbox["xmin"]),
            "ymin": float(bbox["ymin"]),
            "xmax": float(bbox["xmax"]),
            "ymax": float(bbox["ymax"]),
        },
        "aerial": {
            "layer": "Luchtfoto Actueel Ortho 8cm RGB",
            "size_px": int(payload.get("size_px", 4096)),
        },
        "terrain": {
            "ahn_model": payload.get("ahn_model", "DTM"),
            "resolution_m": 0.5,
            "mesh_vertices_per_side": int(payload.get("mesh_vertices", 257)),
        },
        "buildings": {
            "lod": "2.2",
            "clip_mode": payload.get("clip_mode", "centroid"),
            "merge": payload.get("merge", "single"),
        },
        "facade": {
            "variants": int(payload.get("facade_variants", 1)),
            "texture_px": int(payload.get("facade_texture_px", 512)),
            "normal_map": bool(payload.get("facade_normal_map", True)),
            "ground_floor": bool(payload.get("facade_ground_floor", True)),
            "photo_textures": bool(payload.get("facade_photo_textures", True)),
        },
    }

    # Optional sources come as a list of ids from the sources panel. Older
    # callers passing one flag per source still work.
    sys.path.insert(0, str(REPO_ROOT))
    from src.sources import SOURCES, apply_selection

    if "sources" in payload:
        selected = set(payload["sources"] or [])
    else:
        selected = {
            source.id
            for source in SOURCES
            if source.required or payload.get(source.id, True)
        }

    apply_selection(config, selected)
    return config


def run_job(job: Job, preview: bool) -> None:
    """Write the config and drive pipeline.py, streaming its output."""
    work_root = REPO_ROOT / "work" / job.name
    work_root.mkdir(parents=True, exist_ok=True)
    config_path = work_root / "ui_config.json"
    config_path.write_text(json.dumps(job.config, indent=2), encoding="utf-8")

    command = [sys.executable, str(REPO_ROOT / "pipeline.py"), "--config", str(config_path)]
    if preview:
        command.append("--preview")

    job.log(f"$ {' '.join(command)}")
    job.status = "running"

    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        job.log(f"failed to start pipeline: {exc}")
        job.status = "failed"
        job.finished = time.time()
        return

    job.process = process
    assert process.stdout is not None
    for line in process.stdout:
        job.log(line.rstrip("\n"))

    process.wait()
    job.returncode = process.returncode
    job.finished = time.time()

    if job.status == "cancelled":
        job.log("cancelled")
    elif process.returncode == 0:
        job.status = "done"
        job.log("finished")
    else:
        job.status = "failed"
        job.log(f"pipeline exited with code {process.returncode}")


def area_summary(name: str) -> dict | None:
    """Describe a finished area from what is on disk."""
    out_dir = REPO_ROOT / "output" / name
    if not out_dir.is_dir():
        return None

    metadata = None
    metadata_path = out_dir / "metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = None

    validation = None
    validation_path = REPO_ROOT / "work" / name / "validation.json"
    if validation_path.is_file():
        try:
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            validation = None

    files = []
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            files.append(
                {
                    "name": path.name,
                    "size_mb": round(path.stat().st_size / 1e6, 2),
                    "url": f"/output/{name}/{path.name}",
                }
            )

    previews = []
    preview_dir = out_dir / "preview"
    if preview_dir.is_dir():
        for path in sorted(preview_dir.glob("*.png")):
            previews.append(
                {"name": path.stem, "url": f"/output/{name}/preview/{path.name}"}
            )

    return {
        "name": name,
        "modified": datetime.fromtimestamp(
            out_dir.stat().st_mtime, tz=timezone.utc
        ).isoformat(timespec="seconds"),
        "metadata": metadata,
        "validation": validation,
        "files": files,
        "previews": previews,
    }


def list_areas() -> list[dict]:
    root = REPO_ROOT / "output"
    if not root.is_dir():
        return []
    areas = []
    for path in sorted(root.iterdir(), key=lambda p: -p.stat().st_mtime):
        if path.is_dir():
            summary = area_summary(path.name)
            if summary:
                areas.append(summary)
    return areas


class Handler(BaseHTTPRequestHandler):
    server_version = "MapPipelineUI/1.0"

    def log_message(self, fmt: str, *args) -> None:
        # The pipeline's own logging is the interesting output; per-request
        # noise just buries it.
        pass

    # -- helpers ---------------------------------------------------------

    def send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def serve_file(self, path: Path, root: Path) -> None:
        """Serve a file, refusing anything that escapes `root`."""
        try:
            resolved = path.resolve()
            resolved.relative_to(root.resolve())
        except (ValueError, OSError):
            self.send_json({"error": "forbidden"}, status=403)
            return
        if not resolved.is_file():
            self.send_json({"error": "not found"}, status=404)
            return

        types = {
            ".html": "text/html; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".fbx": "application/octet-stream",
            ".txt": "text/plain; charset=utf-8",
            ".pgw": "text/plain; charset=utf-8",
            ".tif": "image/tiff",
        }
        content_type = types.get(resolved.suffix.lower(), "application/octet-stream")
        body = resolved.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if resolved.suffix.lower() in (".fbx", ".tif"):
            self.send_header(
                "Content-Disposition", f'attachment; filename="{resolved.name}"'
            )
        self.end_headers()
        self.wfile.write(body)

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            self.serve_file(UI_DIR / "index.html", UI_DIR)
            return

        if route == "/api/geocode":
            # Proxied rather than called from the page, so the browser never
            # needs a CORS grant from PDOK.
            term = (query.get("q") or [""])[0].strip()
            if not term:
                self.send_json({"results": []})
                return
            # A trailing wildcard turns the exact-match search into a prefix
            # search, which is what typing into a search box needs: plain
            # "Domtoren" matches nothing, "Domtoren*" matches 45 places. The
            # dedicated suggest endpoint does prefix matching too but returns
            # nothing for multi-word queries, and needs a second lookup call to
            # get coordinates.
            if not term.endswith(("*", '"')):
                term += "*"
            url = (
                GEOCODER
                + "?"
                + urllib.parse.urlencode(
                    {
                        "q": term,
                        "rows": 8,
                        "fl": "weergavenaam,type,centroide_rd",
                    }
                )
            )
            try:
                with urllib.request.urlopen(url, timeout=20) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - reported to the page
                self.send_json({"error": str(exc), "results": []}, status=502)
                return

            results = []
            for doc in data.get("response", {}).get("docs", []):
                point = doc.get("centroide_rd", "")
                if not point.startswith("POINT("):
                    continue
                try:
                    x, y = (float(v) for v in point[6:-1].split())
                except ValueError:
                    continue
                results.append(
                    {"label": doc.get("weergavenaam", ""), "type": doc.get("type", ""), "x": x, "y": y}
                )
            self.send_json({"results": results})
            return

        if route == "/api/job":
            job_id = (query.get("id") or [""])[0]
            offset = int((query.get("offset") or ["0"])[0])
            job = JOBS.get(job_id)
            if job is None:
                self.send_json({"error": "unknown job"}, status=404)
                return
            payload = job.snapshot(offset)
            if job.status in ("done", "failed", "cancelled"):
                payload["area"] = area_summary(job.name)
            self.send_json(payload)
            return

        if route == "/api/tile":
            # Fallback for when the browser cannot reach PDOK directly but this
            # process can: behind a corporate proxy, or on a machine where only
            # the pipeline has network access. The page tries the real tile
            # server first and only falls back here.
            try:
                z = int((query.get("z") or ["0"])[0])
                col = int((query.get("col") or ["0"])[0])
                row = int((query.get("row") or ["0"])[0])
            except ValueError:
                self.send_json({"error": "bad tile request"}, status=400)
                return
            if not (0 <= z <= 19 and 0 <= col < 2**z and 0 <= row < 2**z):
                self.send_json({"error": "tile out of range"}, status=400)
                return

            key = f"{z}/{col}/{row}"
            with TILE_CACHE_LOCK:
                cached = TILE_CACHE.get(key)
            if cached is not None:
                self.send_bytes(cached, "image/jpeg")
                return

            url = f"{TILE_BASE}/{z:02d}/{col}/{row}.jpeg"
            try:
                request = urllib.request.Request(
                    url, headers={"User-Agent": "MaptileConverter-UI/1.0"}
                )
                with urllib.request.urlopen(request, timeout=30) as response:
                    body = response.read()
            except Exception as exc:  # noqa: BLE001 - reported as a failed tile
                self.send_json({"error": str(exc)}, status=502)
                return

            with TILE_CACHE_LOCK:
                if len(TILE_CACHE) >= TILE_CACHE_MAX:
                    TILE_CACHE.clear()
                TILE_CACHE[key] = body
            self.send_bytes(body, "image/jpeg")
            return

        if route == "/api/areas":
            self.send_json({"areas": list_areas()})
            return

        if route in ("/api/sources", "/api/health"):
            # The registry decides what a model can be built from; this adds
            # whether each one is answering right now, so an outage is visible
            # before committing to a run rather than after minutes of work.
            sys.path.insert(0, str(REPO_ROOT))
            from src.config import DEFAULTS
            from src.http_util import check_reachable, host_of
            from src.sources import SOURCES, health_targets

            targets = health_targets(DEFAULTS)
            health: dict[str, dict] = {}
            lock = threading.Lock()

            def probe(url: str) -> None:
                ok, detail = check_reachable(url, timeout=6.0)
                with lock:
                    health[url] = {"ok": ok, "detail": detail, "host": host_of(url)}

            # Several sources share a host, so each distinct URL is asked once.
            threads = [
                threading.Thread(target=probe, args=(url,), daemon=True)
                for url in targets
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=12.0)

            listed = []
            for source in SOURCES:
                url = source.probe(DEFAULTS)
                status = health.get(url, {"ok": None, "detail": "not checked", "host": ""})
                listed.append(
                    {
                        "id": source.id,
                        "label": source.label,
                        "provider": source.provider,
                        "contributes": source.contributes,
                        "required": source.required,
                        "note": source.note,
                        "host": status.get("host", ""),
                        "ok": status.get("ok"),
                        "detail": status.get("detail", ""),
                    }
                )

            self.send_json(
                {
                    "sources": listed,
                    "down": sorted(s["id"] for s in listed if s["ok"] is False),
                    "blocking": sorted(
                        s["id"] for s in listed if s["ok"] is False and s["required"]
                    ),
                }
            )
            return

        if route == "/api/estimate":
            def flag(name: str, default: bool = True) -> bool:
                raw = (query.get(name) or [str(default).lower()])[0]
                return raw not in ("0", "false", "no")

            try:
                payload = {
                    "bbox": {
                        "xmin": float((query.get("xmin") or ["0"])[0]),
                        "ymin": float((query.get("ymin") or ["0"])[0]),
                        "xmax": float((query.get("xmax") or ["1000"])[0]),
                        "ymax": float((query.get("ymax") or ["1000"])[0]),
                    },
                    "size_px": int((query.get("size_px") or ["4096"])[0]),
                    "trees": flag("trees"),
                    "water": flag("water"),
                    "land_cover": flag("land_cover"),
                    "furniture": flag("furniture"),
                    "usage": flag("usage"),
                    "preview": flag("preview"),
                }
            except ValueError:
                self.send_json({"error": "bad estimate request"}, status=400)
                return

            speed, samples = measured_speed_factor()
            result = estimate_seconds(payload, speed)
            result["calibrated_from_runs"] = samples
            self.send_json(result)
            return

        if route.startswith("/output/"):
            relative = urllib.parse.unquote(route[len("/output/") :])
            self.serve_file(REPO_ROOT / "output" / relative, REPO_ROOT / "output")
            return

        self.send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path

        if route == "/api/run":
            try:
                payload = self.read_json()
                name = str(payload.get("name", "")).strip()
                if not name:
                    raise ValueError("give the area a name")
                # Keep the name safe as a directory component.
                if not all(c.isalnum() or c in "._-" for c in name):
                    raise ValueError(
                        "the name may only contain letters, digits, dot, dash "
                        "and underscore"
                    )
                config = build_config(payload)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({"error": str(exc)}, status=400)
                return

            speed, _ = measured_speed_factor()
            job = Job(
                uuid.uuid4().hex[:12],
                name,
                config,
                estimate_seconds(payload, speed)["seconds"],
            )
            register(job)
            threading.Thread(
                target=run_job,
                args=(job, bool(payload.get("preview", True))),
                daemon=True,
            ).start()
            self.send_json({"id": job.id, "name": job.name})
            return

        if route == "/api/cancel":
            payload = self.read_json()
            job = JOBS.get(str(payload.get("id", "")))
            if job is None:
                self.send_json({"error": "unknown job"}, status=404)
                return
            job.status = "cancelled"
            if job.process and job.process.poll() is None:
                job.process.terminate()
            self.send_json({"ok": True})
            return

        self.send_json({"error": "not found"}, status=404)


def check_environment() -> list[str]:
    """Report anything that would make a run fail, before the user tries."""
    notes = []
    try:
        import rasterio  # noqa: F401
    except ImportError:
        notes.append("rasterio is not installed: pip install -r requirements.txt")
    try:
        import mapbox_earcut  # noqa: F401
    except ImportError:
        notes.append("mapbox_earcut is not installed: pip install -r requirements.txt")

    if shutil.which("blender") is None:
        try:
            import bpy  # noqa: F401
        except ImportError:
            notes.append(
                "no Blender found: pip install bpy (CPython 3.11), or put "
                "blender on PATH"
            )
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local web UI for the map pipeline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    for note in check_environment():
        print(f"warning: {note}", file=sys.stderr)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"map pipeline UI on http://{args.host}:{args.port}")
    print("press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
