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
import email.utils
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
import uuid
import zipfile
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

# The 3D view. Its file is written by blender/web_export.py into the area's
# work folder; FORMAT there and this number are held equal by a test, so a
# view written for an older page is written again rather than misread.
VIEWER_FORMAT = 3
VIEWER_FILE = "viewer.glb"
VIEWER_STAMP = "viewer.json"
# Writing a view from a saved model takes seconds; this only stops a Blender
# that hangs from holding the queue for ever.
VIEWER_TIMEOUT_S = 30 * 60

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
    # The ground's photo, plus the roofs' own (the "roof textures" stage and
    # its share of the relighting). Kept on this line because estimate_seconds
    # only counts the stages it lists as enabled, and the roofs are the same
    # PDOK imagery. Measured over the Utrecht Dom quarter square kilometre at
    # Standard: 16 s for a 28.6 Mpx roof page, 6 s to fetch and write it and
    # 10 s to relight it, most of both writing PNG; over the square kilometre
    # of config.json, capped at 48 Mpx, 30 s. Draft has no roof page, so this
    # overestimates Draft by a few per cent.
    "aerial":             {"c": 4.0, "a": 12.0, "m": 1.5},
    "surfaces":           {"c": 1.0, "a": 30.0, "m": 2.5},
    "usage":              {"c": 0.5, "a": 5.0,  "m": 0.0},
    "buildings":          {"c": 2.0, "a": 75.0, "m": 0.0},
    # The terrain's edges following the outlines: 48 s over the built-up
    # Bronbeek square kilometre, most of it noding the narrow gardens.
    "terrain_mesh":       {"c": 1.0, "a": 48.0, "m": 0.0},
    # The canopy model was always fetched; finding trees in it rather than only
    # sampling it at the registered points costs about three seconds a square
    # kilometre, measured.
    "trees":              {"c": 1.0, "a": 26.0, "m": 0.0},
    # Seven BGT collections, and the national charging point file, five
    # megabytes whatever the area. 4 s over a 300 m square, measured.
    "furniture":          {"c": 3.0, "a": 6.0,  "m": 0.0},
    # One download per municipality a month (Utrecht is 18 MB), then the
    # faces: drawn in a second, or fetched from Commons at one a second the
    # first time an area needs them and from the cache after that.
    "signs":              {"c": 8.0, "a": 4.0,  "m": 0.0},
    # Lights, road paint and covers: BGT reads and the OpenStreetMap tiles
    # the street objects share. Not measured since the signs left it.
    "traffic":            {"c": 2.0, "a": 8.0,  "m": 0.0},
    # Five collections, and more features than any other BGT stage: 3013
    # separations over one square kilometre of Utrecht. Cheaper than that
    # sounds -- measured at 7.4 s for the demo square, fetch included -- because
    # a swept prism is a handful of quads and none of it needs triangulating.
    "barriers":           {"c": 0.5, "a": 7.0,  "m": 0.0},
    # Two BGT collections, and wegdeel is the big one in a city centre.
    "vehicles":           {"c": 0.5, "a": 14.0, "m": 0.0},
    # One collection, and empty over most of a city.
    "rails":              {"c": 0.3, "a": 4.0,  "m": 0.0},
    # Two collections, plus the surface model when trees are off.
    "structures":         {"c": 0.5, "a": 6.0,  "m": 0.0},
    # Mapillary's search, then checking each photo against the model (at
    # most 250, a third of a second each), reading the walls, and baking the
    # textures. Over 300 m of central Arnhem the whole stage took about two
    # minutes with its photos cached; fetching full-size 360 photos for
    # textures the first time adds minutes on top, which no area can predict.
    "photos":             {"c": 60.0, "a": 240.0, "m": 0.0},
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

        from src.sources import BY_ID

        def on(source_id: str) -> bool:
            source = BY_ID.get(source_id)
            return bool(payload.get(source_id, source.default_on if source else True))

    enabled = {
        "terrain": True,
        "aerial": True,
        "buildings": True,
        "terrain_mesh": True,
        "surfaces": on("water") or on("land_cover"),
        "usage": on("usage"),
        "trees": on("trees"),
        "furniture": on("furniture"),
        "signs": on("signs"),
        "traffic": (
            any(on(i) for i in ("markings", "signals", "covers"))
            if "sources" in payload
            else on("traffic")
        ),
        "barriers": on("barriers"),
        "vehicles": on("vehicles"),
        "rails": on("rails"),
        "structures": on("structures"),
        "photos": on("mapillary"),
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
                "signs": drivers.get("signs", True),
                # Runs from before this stage existed did not do it.
                "traffic": drivers.get("traffic", False),
                "barriers": drivers.get("barriers", True),
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


FINISHED = ("done", "failed", "cancelled", "interrupted")
# How much of a run's log a restarted server reads back.
RESTORED_LOG_LINES = 5000


class Job:
    """One pipeline run.

    A run the page started is also kept on disk, in work/<name>/run.json and
    run.log, so a server started again finds it: one that was running is
    marked interrupted, one that was waiting in the queue is run.
    """

    def __init__(
        self, job_id: str, name: str, config: dict, estimate: float, *, preview: bool = False
    ) -> None:
        self.id = job_id
        self.name = name
        self.config = config
        self.estimate = estimate
        self.preview = preview
        self.lines: list[str] = []
        # queued | starting | running | done | failed | cancelled | interrupted
        self.status = "starting"
        self.returncode: int | None = None
        self.queued_at = time.time()
        self.started = time.time()
        self.finished: float | None = None
        self.process: subprocess.Popen | None = None
        self.lock = threading.Lock()
        self.step = 0
        self.total_steps = 0
        self.stage = "starting"
        # Set for runs the page started; a Job made in a test stays in memory.
        self.persist = False
        self._log_file = None

    @property
    def folder(self) -> Path:
        return REPO_ROOT / "work" / self.name

    def save(self) -> None:
        """Write the run's record beside its work files."""
        if not self.persist:
            return
        with self.lock:
            record = {
                "id": self.id, "name": self.name, "status": self.status,
                "returncode": self.returncode, "estimate": self.estimate,
                "preview": self.preview, "queued_at": self.queued_at,
                "started": self.started, "finished": self.finished,
                "step": self.step, "total_steps": self.total_steps, "stage": self.stage,
            }
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            partial = self.folder / "run.json.part"
            partial.write_text(json.dumps(record, indent=2), encoding="utf-8")
            partial.replace(self.folder / "run.json")
        except OSError:
            pass

    def open_log(self) -> None:
        if self.persist:
            try:
                self.folder.mkdir(parents=True, exist_ok=True)
                self._log_file = (self.folder / "run.log").open("w", encoding="utf-8")
            except OSError:
                self._log_file = None

    def close_log(self) -> None:
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def log(self, line: str) -> None:
        with self.lock:
            self.lines.append(line)
            step = self.step
            self._track_progress(line)
            changed = self.step != step
            if self._log_file is not None:
                try:
                    self._log_file.write(line + "\n")
                    self._log_file.flush()
                except OSError:
                    pass
        if changed:
            self.save()

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
        if self.status == "queued":
            elapsed = 0.0
        if self.status in FINISHED:
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
            "style": self.config.get("style"),
            "detail": self.config.get("detail"),
            "queued_at": self.queued_at,
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
        # Forget the oldest finished runs; a waiting or running one stays.
        while len(JOBS_ORDER) > MAX_JOBS:
            old = next(
                (i for i in JOBS_ORDER if JOBS.get(i) is None or JOBS[i].status in FINISHED),
                None,
            )
            if old is None:
                break
            JOBS_ORDER.remove(old)
            JOBS.pop(old, None)


# The build queue: one run at a time, in the order they were asked for. Each
# build loads a whole model into memory and Blender, and two at once would
# only both go slower.
QUEUE: list[str] = []
QUEUE_LOCK = threading.Condition()
_WORKER: threading.Thread | None = None


def active_job(name: str) -> Job | None:
    """A run of this area that is waiting or going, if there is one."""
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    return next(
        (j for j in jobs if j.name == name and j.status in ("queued", "starting", "running")), None
    )


def submit(job: Job) -> None:
    """Put a run in the queue; the worker starts it when its turn comes."""
    job.persist = True
    job.status = "queued"
    job.queued_at = time.time()
    job.folder.mkdir(parents=True, exist_ok=True)
    (job.folder / "ui_config.json").write_text(json.dumps(job.config, indent=2), encoding="utf-8")
    register(job)
    job.save()
    with QUEUE_LOCK:
        QUEUE.append(job.id)
        QUEUE_LOCK.notify()
    ensure_worker()


def ensure_worker() -> None:
    global _WORKER
    with QUEUE_LOCK:
        if _WORKER is not None and _WORKER.is_alive():
            return
        _WORKER = threading.Thread(target=_work, name="build-queue", daemon=True)
        _WORKER.start()


def _work() -> None:
    while True:
        with QUEUE_LOCK:
            while not QUEUE:
                QUEUE_LOCK.wait()
            job_id = QUEUE.pop(0)
        job = JOBS.get(job_id)
        if job is None or job.status != "queued":
            continue
        try:
            run_job(job, job.preview)
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop the queue
            job.log(f"the run stopped: {type(exc).__name__}: {exc}")
            job.status = "failed"
            job.finished = time.time()
            job.save()


def cancel(job: Job) -> None:
    """Stop a run, or take it out of the queue before it starts."""
    with QUEUE_LOCK:
        waiting = job.id in QUEUE
        if waiting:
            QUEUE.remove(job.id)
    previous = job.status
    job.status = "cancelled"
    if waiting or previous == "queued":
        job.finished = time.time()
        job.save()
        return
    if job.process:
        stop_process(job.process)


def stop_process(process: subprocess.Popen) -> None:
    """Stop a pipeline and what it started.

    Blender runs as the pipeline's child, so stopping the pipeline alone
    would leave Blender building on, into the folder the next run of that
    area writes to. The pipeline runs in a process group of its own (see
    run_job), and the whole group is stopped.
    """
    if process.poll() is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(process.pid)], capture_output=True, check=False
            )
    except (ProcessLookupError, PermissionError, OSError):
        process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        process.kill()


def stop_running() -> None:
    """On the way out: stop the run that is building and mark it interrupted.

    The next server offers it again with *Opnieuw starten*; runs still
    waiting stay queued on disk and start by themselves.
    """
    with JOBS_LOCK:
        running = [job for job in JOBS.values() if job.status == "running"]
    for job in running:
        job.status = "interrupted"
        job.finished = time.time()
        job.save()
        if job.process:
            stop_process(job.process)


def restore_jobs() -> int:
    """Pick up the runs a previous server left on disk.

    A run that was going when the server stopped did not finish: its
    pipeline was the server's child and lost its output, so it is marked
    interrupted, with its log kept. A run still waiting is queued again.
    Returns how many were queued again.
    """
    records = []
    for path in sorted((REPO_ROOT / "work").glob("*/run.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict) and record.get("id") and record.get("name") == path.parent.name:
            records.append(record)
    records.sort(key=lambda r: r.get("queued_at") or 0)

    requeued = 0
    for record in records[-MAX_JOBS:]:
        if record["id"] in JOBS:
            continue
        folder = REPO_ROOT / "work" / record["name"]
        try:
            config = json.loads((folder / "ui_config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            config = {}
        job = Job(record["id"], record["name"], config, float(record.get("estimate") or 0),
                  preview=bool(record.get("preview")))
        job.persist = True
        for key in ("returncode", "queued_at", "started", "finished", "step", "total_steps", "stage"):
            if record.get(key) is not None:
                setattr(job, key, record[key])
        try:
            with (folder / "run.log").open(encoding="utf-8", errors="replace") as handle:
                job.lines = [line.rstrip("\n") for line in handle][-RESTORED_LOG_LINES:]
        except OSError:
            pass
        status = record.get("status", "failed")
        if status in ("starting", "running"):
            job.status = "interrupted"
            job.finished = job.finished or time.time()
            job.lines.append("interrupted: the server stopped while this model was being built")
        else:
            job.status = status
        register(job)
        if job.status == "queued":
            with QUEUE_LOCK:
                QUEUE.append(job.id)
                QUEUE_LOCK.notify()
            requeued += 1
        else:
            job.save()
    if requeued:
        ensure_worker()
    return requeued


def _facade_variety(payload: dict) -> dict:
    """Turn the form's buildings-per-wall budget into the facade settings.

    The form asks the question the user can answer -- how many buildings may
    share a wall -- rather than how many textures to generate, because the
    second depends on what is actually in the area. 0 is the off switch.

    `facade_looks_per_era` is still read for a caller written against the
    older form.
    """
    budget = payload.get("facade_buildings_per_look")
    if budget is None:
        looks = int(payload.get("facade_looks_per_era", 12))
        return {"looks_per_era": max(1, looks)}
    budget = int(budget)
    if budget <= 0:
        return {"looks_per_era": 1}
    return {"looks_per_era": 12, "buildings_per_look": budget}


# Payload keys a caller may still set one by one, and where each goes. The
# form sends a detail level instead; these are for scripts and older callers,
# and win over the level when present.
_OVERRIDES = {
    "size_px": ("aerial", "size_px", int),
    "terrain_grid_m": ("terrain", "grid_m", float),
    "facade_texture_px": ("facade", "texture_px", int),
    "facade_normal_map": ("facade", "normal_map", bool),
    "facade_variants": ("facade", "variants", int),
    "facade_ground_floor": ("facade", "ground_floor", bool),
    "facade_photo_textures": ("facade", "photo_textures", bool),
    "clip_mode": ("buildings", "clip_mode", str),
    "deshadow": ("aerial", "deshadow", float),
    "tree_ground_shade": ("trees", "ground_shade", float),
    "car_occupancy": ("vehicles", "car_occupancy", float),
    "collision_tree_crowns": ("export", "collision_tree_crowns", bool),
    "collision_thickness_m": ("barriers", "collision_thickness_m", float),
    # The layered ground's layout: "quads" (the grid) or "ribcage".
    "mesher": ("layered", "mesher", str),
}


def build_config(payload: dict) -> dict:
    """Turn the form payload into a pipeline config.

    The form sends an area, a style, a detail level, the layers wanted and
    whether to add collision shapes. Everything else is the pipeline's default, which is
    the point: the settings that used to be sliders either follow from the
    detail level or have one right answer.
    """
    bbox = payload["bbox"]
    config: dict = {
        "name": payload["name"],
        "bbox": {
            "crs": "EPSG:28992",
            "xmin": float(bbox["xmin"]),
            "ymin": float(bbox["ymin"]),
            "xmax": float(bbox["xmax"]),
            "ymax": float(bbox["ymax"]),
        },
        # Resolved by load_config against DETAIL_PRESETS and STYLE_PRESETS.
        "detail": str(payload.get("detail", "standard")),
        "style": str(payload.get("style", "layered")),
        "export": {
            # Physics proxies for a simulation. Off by default: they add to the
            # FBX and most people want a picture.
            "collision": bool(payload.get("collision", False)),
        },
    }
    for key, (section, field, kind) in _OVERRIDES.items():
        if key in payload and payload[key] is not None:
            config.setdefault(section, {})[field] = kind(payload[key])
    if "facade_buildings_per_look" in payload or "facade_looks_per_era" in payload:
        config.setdefault("facade", {}).update(_facade_variety(payload))

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
            if source.required or payload.get(source.id, source.default_on)
        }

    apply_selection(config, selected)

    # Photos dropped on the page for this area are used whenever there are
    # any; the Mapillary layer is a source of its own.
    from src.photo_folder import folder_for, photos_in

    name = str(payload.get("name", ""))
    if name and area_name_ok(name) and photos_in(folder_for(REPO_ROOT, name)):
        config.setdefault("photos", {})["folder"] = str(folder_for(REPO_ROOT, name))
    if payload.get("photo_textures") in ("own", "all", "off"):
        config.setdefault("photos", {})["textures"] = payload["photo_textures"]
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

    job.started = time.time()
    job.status = "running"
    job.open_log()
    job.log(f"$ {' '.join(command)}")
    job.save()

    try:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            # A group of its own, so stopping the run stops Blender too, and
            # Ctrl+C on the server reaches the run only through stop_running.
            **({"start_new_session": True} if os.name == "posix"
               else {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}),
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        job.log(f"failed to start pipeline: {exc}")
        job.status = "failed"
        job.finished = time.time()
        job.close_log()
        job.save()
        return

    job.process = process
    assert process.stdout is not None
    for line in process.stdout:
        job.log(line.rstrip("\n"))

    process.wait()
    job.returncode = process.returncode
    job.finished = time.time()

    if job.status in ("cancelled", "interrupted"):
        job.log(job.status)
    elif process.returncode == 0:
        job.status = "done"
        job.log("finished")
    else:
        job.status = "failed"
        job.log(f"pipeline exited with code {process.returncode}")
    job.close_log()
    job.save()


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

    photo_checks = []
    checks_dir = out_dir / "photo_checks"
    if checks_dir.is_dir():
        for path in sorted(checks_dir.glob("*.jpg")):
            photo_checks.append(
                {"name": path.stem, "url": f"/output/{name}/photo_checks/{path.name}"}
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
        "photo_checks": photo_checks,
        "settings": area_settings(name),
        # Whether the 3D view can be opened, asked without starting anything:
        # "unavailable" is the one state where it cannot.
        "view": viewer_state(name, start=False)["state"],
    }


def area_settings(name: str) -> dict:
    """What an area was built with, so the page can build it again.

    The page's own runs leave their config in `work/<name>/ui_config.json`;
    a run started from the command line leaves only the scene, which still
    names the style. Whatever is not known is left out.
    """
    work = REPO_ROOT / "work" / name
    settings: dict = {}
    for source in ("ui_config.json", "scene.json"):
        path = work / source
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for key in ("style", "detail"):
            if isinstance(data.get(key), str):
                settings.setdefault(key, data[key])
        export = data.get("export")
        if isinstance(export, dict) and "collision" in export:
            settings.setdefault("collision", bool(export["collision"]))
        layered = data.get("layered")
        if isinstance(layered, dict) and isinstance(layered.get("mesher"), str):
            settings.setdefault("mesher", layered["mesher"])
    return settings


def list_jobs() -> list[dict]:
    """Every run this server knows of, newest first, without their logs.

    For a page opened, or reloaded, while a build is running: it picks the run
    up from here and follows it with `/api/job`.
    """
    with JOBS_LOCK:
        jobs = [JOBS[i] for i in reversed(JOBS_ORDER) if i in JOBS]
    out = []
    for job in jobs:
        snap = job.snapshot(10**9)
        snap.pop("lines", None)
        out.append(snap)
    return out


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


# ---------------------------------------------------------------------------
# Removing a model, and handing one over whole.
# ---------------------------------------------------------------------------

# Already compressed: deflating them again costs time and saves nothing.
STORED_SUFFIXES = {".png", ".jpg", ".jpeg", ".blend", ".zip", ".glb"}
UNITY_PACKAGE = "com.terreinlab.importer"


def delete_area(name: str) -> tuple[int, str] | None:
    """Remove an area's model and its work files.

    Returns the HTTP status and why not, or None when it is gone.
    """
    if not area_name_ok(name):
        return 400, "bad model name"
    if active_job(name):
        return 409, "this model is being built or waits in the queue"
    with VIEWER_LOCK:
        if VIEWER_PREP.get(name, {}).get("status") == "running":
            return 409, "its 3D view is being written; try again in a moment"
    removed = False
    for root in ("output", "work"):
        base = (REPO_ROOT / root).resolve()
        folder = (REPO_ROOT / root / name).resolve()
        if folder.parent != base or not folder.is_dir():
            continue
        shutil.rmtree(folder)
        removed = True
    with JOBS_LOCK:
        for job_id in [i for i in JOBS_ORDER if JOBS.get(i) and JOBS[i].name == name]:
            JOBS_ORDER.remove(job_id)
            JOBS.pop(job_id, None)
    with VIEWER_LOCK:
        VIEWER_PREP.pop(name, None)
    return None if removed else (404, "there is no model called " + name)


def write_zip(folder: Path, top: str, handle) -> int:
    """Every file under `folder` into a zip on `handle`, under `top/`."""
    count = 0
    with zipfile.ZipFile(handle, "w") as archive:
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            stored = path.suffix.lower() in STORED_SUFFIXES
            archive.write(
                path,
                f"{top}/{path.relative_to(folder).as_posix()}",
                compress_type=zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED,
                compresslevel=None if stored else 6,
            )
            count += 1
    return count


# ---------------------------------------------------------------------------
# The 3D view.
#
# A build writes it as it finishes. A model built before there was a view, or
# edited and saved in Blender since, gets one written from its .blend the
# first time it is opened: the stamp beside the view says which save of the
# .blend it shows, so an edit is noticed without anyone having to say so.
# ---------------------------------------------------------------------------

# Per area: a view being written, or the last attempt that failed and why.
VIEWER_PREP: dict[str, dict] = {}
VIEWER_LOCK = threading.Lock()
# One Blender at a time: each loads a whole model.
VIEWER_QUEUE = threading.Lock()


def area_name_ok(name: str) -> bool:
    """A name that can only ever be one folder inside output/ and work/."""
    return (
        bool(name)
        and name not in (".", "..")
        and all(c.isalnum() or c in "._-" for c in name)
    )


def model_blend(name: str) -> Path | None:
    """The saved model of an area: model.blend, or else the newest .blend."""
    out_dir = REPO_ROOT / "output" / name
    preferred = out_dir / "model.blend"
    if preferred.is_file():
        return preferred
    found = sorted(out_dir.glob("*.blend"), key=lambda p: p.stat().st_mtime, reverse=True)
    return found[0] if found else None


def _read_stamp(path: Path) -> dict | None:
    try:
        stamp = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return stamp if isinstance(stamp, dict) else None


def _blend_key(blend: Path) -> tuple[int, int] | None:
    try:
        stat = blend.stat()
    except OSError:
        return None
    return int(stat.st_size), int(stat.st_mtime_ns)


def view_is_current(stamp: dict | None, blend: Path | None) -> bool:
    """Whether a view shows the saved model as it is now, in today's layout."""
    if not stamp or stamp.get("format") != VIEWER_FORMAT:
        return False
    if blend is None:
        # Nothing to write a newer one from, so this is the model.
        return True
    return (
        stamp.get("source") == blend.name
        and (stamp.get("source_size"), stamp.get("source_mtime_ns")) == _blend_key(blend)
    )


def area_is_building(name: str) -> bool:
    with JOBS_LOCK:
        jobs = list(JOBS.values())
    return any(j.name == name and j.status in ("starting", "running") for j in jobs)


def _error_line(stderr: str) -> str:
    """The line of a Blender traceback that says what went wrong."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    for line in reversed(lines):
        if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit)\b", line):
            return line[:300]
    return lines[-1][:300] if lines else ""


def prepare_view(name: str, blend: Path, key: tuple[int, int] | None) -> None:
    """Write an area's 3D view from its .blend, in a Blender of its own."""
    glb = REPO_ROOT / "work" / name / VIEWER_FILE
    error = ""
    with VIEWER_QUEUE:
        try:
            from pipeline import blender_command

            _, command = blender_command(
                "web_export.py", ["--blend", str(blend), "--out", str(glb)], None
            )
            result = subprocess.run(
                command, cwd=str(REPO_ROOT), capture_output=True, text=True,
                timeout=VIEWER_TIMEOUT_S,
            )
            stamp = _read_stamp(glb.with_name(VIEWER_STAMP))
            # Blender in background mode exits 0 when its script raises, so
            # the file and its stamp are what count.
            if result.returncode != 0 or not glb.is_file() or not stamp or (
                stamp.get("format") != VIEWER_FORMAT
            ):
                error = _error_line(result.stderr) or (
                    f"Blender stopped with code {result.returncode} without writing the view"
                )
        except subprocess.TimeoutExpired:
            error = f"Blender took longer than {VIEWER_TIMEOUT_S // 60} minutes"
        except SystemExit as exc:
            # find_blender's way of saying there is no Blender at all.
            error = str(exc).splitlines()[0] if str(exc) else "no Blender found"
        except Exception as exc:  # noqa: BLE001 - shown on the page
            error = f"{type(exc).__name__}: {exc}"

    with VIEWER_LOCK:
        VIEWER_PREP[name] = {
            "status": "failed" if error else "done",
            "error": error,
            "key": key,
            "finished": time.time(),
        }


def viewer_state(name: str, *, retry: bool = False, start: bool = True) -> dict:
    """Where an area's 3D view stands, starting to write it when it is due.

    `ready` comes with the file's address and where its textures are;
    `preparing` while Blender writes it; `building` while the area itself is
    being built and has no view yet; `failed` with the reason, until asked
    to `retry`; `unavailable` when there is no saved model to write one from.
    """
    out_dir = REPO_ROOT / "output" / name
    if not out_dir.is_dir():
        return {"state": "unavailable", "error": f"there is no model called {name}"}

    glb = REPO_ROOT / "work" / name / VIEWER_FILE
    blend = model_blend(name)
    stamp = _read_stamp(glb.with_name(VIEWER_STAMP)) if glb.is_file() else None
    current = glb.is_file() and view_is_current(stamp, blend)
    readable = glb.is_file() and bool(stamp) and stamp.get("format") == VIEWER_FORMAT
    building = area_is_building(name)

    if current or (readable and building):
        quoted = urllib.parse.quote(name)
        return {
            "state": "ready",
            "url": f"/viewer/{quoted}/{VIEWER_FILE}?v={glb.stat().st_mtime_ns}",
            "textures": f"/output/{quoted}/",
            "bytes": glb.stat().st_size,
            # Shown while the area is rebuilt: the view of the model before.
            "stale": not current,
            "stats": {k: stamp.get(k) for k in ("triangles", "objects", "layers")},
        }
    if building:
        return {"state": "building"}
    if blend is None:
        return {
            "state": "unavailable",
            "error": "this model has no .blend to make a 3D view from",
        }

    key = _blend_key(blend)
    with VIEWER_LOCK:
        prep = VIEWER_PREP.get(name)
        if prep and prep["status"] == "running":
            return {"state": "preparing", "elapsed": round(time.time() - prep["started"], 1)}
        if prep and prep["key"] == key and not retry:
            if prep["status"] == "failed":
                return {"state": "failed", "error": prep["error"]}
            # Written for this very save and still not current: writing it
            # again would only go round in a circle.
            return {
                "state": "failed",
                "error": "the view written does not match the saved model; try again",
            }
        if not start:
            return {"state": "absent"}
        VIEWER_PREP[name] = {"status": "running", "started": time.time(), "key": key}
    threading.Thread(target=prepare_view, args=(name, blend, key), daemon=True).start()
    return {"state": "preparing", "elapsed": 0.0}


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
        """Serve a file, refusing anything that escapes `root`.

        Streamed rather than read whole: the 3D view and the aerial are tens
        of megabytes. Every response carries its modification time and asks
        the browser to check back, so opening a model again costs a "not
        modified" per file instead of the whole download.
        """
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
            ".js": "text/javascript; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".fbx": "application/octet-stream",
            ".glb": "model/gltf-binary",
            ".txt": "text/plain; charset=utf-8",
            ".pgw": "text/plain; charset=utf-8",
            ".tif": "image/tiff",
            ".svg": "image/svg+xml",
        }
        content_type = types.get(resolved.suffix.lower(), "application/octet-stream")
        stat = resolved.stat()
        modified = email.utils.formatdate(stat.st_mtime, usegmt=True)

        since = self.headers.get("If-Modified-Since")
        if since:
            try:
                unchanged = int(stat.st_mtime) <= email.utils.parsedate_to_datetime(since).timestamp()
            except (TypeError, ValueError, IndexError):
                unchanged = False
            if unchanged:
                self.send_response(304)
                self.send_header("Last-Modified", modified)
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(stat.st_size))
        self.send_header("Last-Modified", modified)
        self.send_header("Cache-Control", "no-cache")
        if resolved.suffix.lower() in (".fbx", ".tif"):
            self.send_header(
                "Content-Disposition", f'attachment; filename="{resolved.name}"'
            )
        self.end_headers()
        try:
            with resolved.open("rb") as handle:
                shutil.copyfileobj(handle, self.wfile, 1 << 20)
        except (BrokenPipeError, ConnectionResetError):
            # The page went elsewhere mid-download, which is allowed.
            pass

    # -- routes ----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if route in ("/", "/index.html"):
            self.serve_file(UI_DIR / "index.html", UI_DIR)
            return
        if route in ("/icon.svg", "/logo-dark.svg", "/logo-light.svg", "/viewer.js"):
            self.serve_file(UI_DIR / route.lstrip("/"), UI_DIR)
            return

        if route.startswith("/api/photos") or route == "/api/mapillary":
            self.photos_get(route, query)
            return

        if route == "/api/viewer":
            # Where an area's 3D view stands; asking starts writing it when
            # the model has none, or only one of an older save.
            name = (query.get("name") or [""])[0]
            if not area_name_ok(name):
                self.send_json({"error": "bad model name"}, status=400)
                return
            retry = (query.get("retry") or ["0"])[0] not in ("", "0", "false", "no")
            self.send_json(viewer_state(name, retry=retry))
            return

        if route.startswith("/viewer/"):
            # The view itself, from the work folder: only that one file.
            parts = urllib.parse.unquote(route[len("/viewer/") :]).split("/")
            if len(parts) == 2 and parts[1] == VIEWER_FILE and area_name_ok(parts[0]):
                self.serve_file(
                    REPO_ROOT / "work" / parts[0] / VIEWER_FILE, REPO_ROOT / "work"
                )
                return
            self.send_json({"error": "not found"}, status=404)
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
            if job.status in FINISHED:
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
                    url, headers={"User-Agent": "TerreinLab-UI/1.0"}
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

        if route == "/api/jobs":
            self.send_json({"jobs": list_jobs()})
            return

        if route == "/api/zip":
            # A whole model as one download, folder and all, to pass on.
            name = (query.get("name") or [""])[0]
            folder = REPO_ROOT / "output" / name
            if not area_name_ok(name) or not folder.is_dir():
                self.send_json({"error": "no such model"}, status=404)
                return
            self.send_zip(folder, name, f"{name}.zip")
            return

        if route == "/api/unity-package.zip":
            folder = REPO_ROOT / "unity" / UNITY_PACKAGE
            if not (folder / "package.json").is_file():
                self.send_json({"error": "the Unity package is not in this copy"}, status=404)
                return
            self.send_zip(folder, UNITY_PACKAGE, f"{UNITY_PACKAGE}.zip")
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
                # A source can have a second service carrying the same data, so
                # it is only down when none of them answers. Reporting the first
                # one as decisive would show a blocking outage for a run that
                # would have finished on the fallback.
                urls = source.probes(DEFAULTS)
                answering = next(
                    (u for u in urls if health.get(u, {}).get("ok")), None
                )
                chosen = answering or (urls[0] if urls else "")
                status = health.get(
                    chosen, {"ok": None, "detail": "not checked", "host": ""}
                )
                detail = status.get("detail", "")
                if answering and len(urls) > 1 and answering != urls[0]:
                    detail = f"{detail} (via the backup service)"
                # A service can answer and still be unusable to us: Mapillary
                # without a token is that, and it is something the person can
                # fix, unlike an outage.
                missing = ""
                if source.id == "mapillary":
                    from src.mapillary import MissingToken, find_token

                    try:
                        find_token()
                    except MissingToken:
                        missing = "token"
                listed.append(
                    {
                        "id": source.id,
                        "label": source.label,
                        "provider": source.provider,
                        "contributes": source.contributes,
                        "required": source.required,
                        "note": source.note,
                        "default_on": source.default_on,
                        "missing": missing,
                        "host": status.get("host", ""),
                        "ok": status.get("ok"),
                        "detail": detail,
                        "degraded": bool(
                            answering and len(urls) > 1 and answering != urls[0]
                        ),
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
                    "signs": flag("signs"),
                    "traffic": flag("traffic"),
                    "barriers": flag("barriers"),
                    "usage": flag("usage"),
                    # The page sends these too; leaving them out here counted
                    # every one of them in, switched off or not.
                    "vehicles": flag("vehicles"),
                    "rails": flag("rails"),
                    "structures": flag("structures"),
                    "mapillary": flag("mapillary", False),
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

    # -- photos ----------------------------------------------------------

    def _photo_folder(self, area: str):
        from src.photo_folder import folder_for

        if not area_name_ok(area):
            raise ValueError("give the area a name of letters, digits, dots, dashes or underscores")
        return folder_for(REPO_ROOT, area)

    def photos_get(self, route: str, query: dict) -> None:
        from src import photo_folder

        try:
            if route == "/api/mapillary":
                sys.path.insert(0, str(REPO_ROOT))
                from src.geo import BBox

                box = BBox(*(float((query.get(k) or ["0"])[0]) for k in ("xmin", "ymin", "xmax", "ymax")))
                if box.width > 2500 or box.height > 2500:
                    self.send_json({"error": "zoom in: at most 2.5 km across"}, status=400)
                    return
                found = photo_folder.mapillary_points(box, REPO_ROOT / "work" / "_mapillary_preview")
                self.send_json({**found, "count": found["total"]})
                return
            folder = self._photo_folder((query.get("area") or [""])[0])
            name = (query.get("name") or [""])[0]
            if route == "/api/photos":
                self.send_json({"photos": photo_folder.describe(folder)})
            elif route == "/api/photos/thumb":
                self.send_bytes(photo_folder.thumbnail(folder, name), "image/jpeg")
            elif route == "/api/photos/file":
                self.serve_file(folder / photo_folder.safe_name(name), folder)
            else:
                self.send_json({"error": "not found"}, status=404)
        except FileNotFoundError:
            self.send_json({"error": "no such photo"}, status=404)
        except ValueError as exc:
            self.send_json({"error": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 - reported to the page, not swallowed
            self.send_json({"error": str(exc)}, status=502)

    def photos_post(self, route: str, query: dict) -> None:
        from src import photo_folder

        try:
            if route == "/api/photos/upload":
                folder = self._photo_folder((query.get("area") or [""])[0])
                length = int(self.headers.get("Content-Length", 0))
                if length > photo_folder.MAX_UPLOAD_BYTES:
                    raise ValueError("a photo may be at most 80 MB")
                data = self.rfile.read(length)
                stored = photo_folder.save_upload(folder, (query.get("name") or ["photo.jpg"])[0], data)
                self.send_json({"ok": True, "name": stored})
                return
            payload = self.read_json()
            folder = self._photo_folder(str(payload.get("area", "")))
            name = str(payload.get("name", ""))
            if route == "/api/photos/pose":
                heading = payload.get("heading")
                side = photo_folder.set_pose(
                    folder, name, x=float(payload["x"]), y=float(payload["y"]),
                    heading=None if heading is None else float(heading),
                )
            elif route == "/api/photos/pin":
                if payload.get("clear"):
                    side = photo_folder.clear_pin(folder, name)
                else:
                    side = photo_folder.set_pin(
                        folder, name, corners=payload["corners"],
                        x=float(payload["x"]), y=float(payload["y"]),
                    )
            elif route == "/api/photos/delete":
                photo_folder.delete_photo(folder, name)
                side = {}
            else:
                self.send_json({"error": "not found"}, status=404)
                return
            self.send_json({"ok": True, "sidecar": side})
        except FileNotFoundError:
            self.send_json({"error": "no such photo"}, status=404)
        except (ValueError, KeyError, TypeError) as exc:
            self.send_json({"error": str(exc)}, status=400)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        if route.startswith("/api/photos/"):
            self.photos_post(route, urllib.parse.parse_qs(parsed.query))
            return

        if route == "/api/run":
            try:
                payload = self.read_json()
                name = str(payload.get("name", "")).strip()
                if not name:
                    raise ValueError("give the area a name")
                # Keep the name safe as a directory component: one folder in
                # output/ and work/, never ".." or "." either.
                if not area_name_ok(name):
                    raise ValueError(
                        "the name may only contain letters, digits, dot, dash "
                        "and underscore, and may not be only dots"
                    )
                config = build_config(payload)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({"error": str(exc)}, status=400)
                return
            if active_job(name):
                self.send_json(
                    {"error": f"{name} is already being built or waits in the queue"}, status=409
                )
                return

            speed, _ = measured_speed_factor()
            job = Job(
                uuid.uuid4().hex[:12],
                name,
                config,
                estimate_seconds(payload, speed)["seconds"],
                preview=bool(payload.get("preview", True)),
            )
            submit(job)
            self.send_json({"id": job.id, "name": job.name, "status": job.status})
            return

        if route == "/api/rerun":
            # A run again with what it was asked for: after an interruption,
            # or a failure worth another go.
            payload = self.read_json()
            old = JOBS.get(str(payload.get("id", "")))
            if old is None or not old.config.get("bbox"):
                self.send_json({"error": "unknown job"}, status=404)
                return
            if active_job(old.name):
                self.send_json(
                    {"error": f"{old.name} is already being built or waits in the queue"}, status=409
                )
                return
            job = Job(uuid.uuid4().hex[:12], old.name, old.config, old.estimate, preview=old.preview)
            submit(job)
            self.send_json({"id": job.id, "name": job.name, "status": job.status})
            return

        if route == "/api/cancel":
            payload = self.read_json()
            job = JOBS.get(str(payload.get("id", "")))
            if job is None:
                self.send_json({"error": "unknown job"}, status=404)
                return
            cancel(job)
            self.send_json({"ok": True})
            return

        if route == "/api/delete":
            payload = self.read_json()
            refused = delete_area(str(payload.get("name", "")))
            if refused:
                self.send_json({"error": refused[1]}, status=refused[0])
                return
            self.send_json({"ok": True})
            return

        self.send_json({"error": "not found"}, status=404)

    def send_zip(self, folder: Path, top: str, filename: str) -> None:
        """A folder as one zip download, written aside first so the browser
        knows its size and can show how far it is."""
        with tempfile.TemporaryFile() as handle:
            write_zip(folder, top, handle)
            size = handle.tell()
            handle.seek(0)
            self.send_response(200)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            try:
                shutil.copyfileobj(handle, self.wfile, 1 << 20)
            except (BrokenPipeError, ConnectionResetError):
                pass


def check_environment() -> list[str]:
    """Report anything that would make a run fail, before the user tries."""
    from src.dependencies import install_command, missing_dependencies

    missing = missing_dependencies()
    if missing:
        # pipeline.py imports these, so the Blender search below cannot run.
        return [*missing, f"install them with: {install_command()}"]
    notes = []

    # Not shutil.which: on macOS Blender is an .app bundle, so its executable
    # is never on PATH however it was installed.
    from pipeline import pip_bpy_version, search_for_blender

    # The pip module by its package record: importing it here would load all
    # of Blender into the web server.
    if not search_for_blender() and pip_bpy_version() is None:
        notes.append(
            "no Blender found: install Blender 5 from blender.org, or "
            "pip install -r requirements.txt (CPython 3.13)"
        )
    return notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TerreinLab: the local web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--open",
        action="store_true",
        help="open the page in a browser once the server is listening",
    )
    args = parser.parse_args(argv)

    for note in check_environment():
        print(f"warning: {note}", file=sys.stderr)
    requeued = restore_jobs()
    if requeued:
        print(f"{requeued} model(s) still in the queue from before; building them now")

    # Bind before opening the browser, or the page races the socket and the
    # first thing a new user sees is a connection error.
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    if args.open:
        import threading
        import webbrowser

        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    print(f"TerreinLab on {url}")
    print("press Ctrl+C to stop")

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    # A plain `kill` stops the server the way Ctrl+C does, run included.
    signal.signal(signal.SIGTERM, stop)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        stop_running()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
