"""Road centrelines from the Nationaal Wegenbestand (NWB-Wegen).

The BGT draws every road as a mosaic of surfaces -- carriageway, footway,
cycle path, parking bay -- with no line down the middle of any of them. A road
mesh that an editor can work with needs that line: it is what decides where
the edge loops go across a street. NWB is Rijkswaterstaat's national road
network, published monthly under CC-0, and for municipal roads its geometry is
itself drawn from the BGT, which is why it lines up. Measured over Bronbeek,
inside the area the centreline sits on the carriageway at 92% of 2 m samples,
12 cm from the middle of it at the median.

Each road section (``wegvak``) runs from one junction to the next and carries
both junction ids, so the network's topology arrives explicit rather than
having to be inferred from coincident endpoints.

NWB has no widths. The kerbs come from the BGT; the centreline only decides
where the cross-sections are cut.
"""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geo import BBox
from .http_util import get_with_retry, short_error

LOG = logging.getLogger(__name__)

NWB_URL = (
    "https://api.pdok.nl/rws/nationaal-wegenbestand-wegen/ogc/v1/"
    "collections/wegvakken/items"
)
RD_URI = "http://www.opengis.net/def/crs/EPSG/0/28992"


@dataclass
class Centreline:
    """One NWB road section, junction to junction."""

    points: np.ndarray  # (n, 2) RD
    section_id: int
    start_junction: int
    end_junction: int
    name: str
    # BST code: RB carriageway, FP cycle path, PAR parallel road, HR main
    # carriageway of a dual road, and a handful more. "skeleton" for one the
    # road module drew itself where NWB has nothing.
    kind: str
    # Which connected network a drawn centreline belongs to; 0 for NWB.
    group: int = 0

    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())


@dataclass
class Network:
    lines: list[Centreline] = field(default_factory=list)
    source: str = ""

    def __len__(self) -> int:
        return len(self.lines)

    def degree(self) -> dict[int, int]:
        """How many sections meet at each junction."""
        out: dict[int, int] = {}
        for line in self.lines:
            for junction in (line.start_junction, line.end_junction):
                out[junction] = out.get(junction, 0) + 1
        return out


def parse_features(features: list[dict]) -> Network:
    """GeoJSON features from the wegvakken collection into centrelines."""
    network = Network(source="NWB-Wegen wegvakken")
    for feature in features:
        geometry = feature.get("geometry") or {}
        props = feature.get("properties") or {}
        kind = geometry.get("type")
        if kind == "LineString":
            parts = [geometry.get("coordinates") or []]
        elif kind == "MultiLineString":
            parts = geometry.get("coordinates") or []
        else:
            continue
        # A section is one line; the multi form is a container that almost
        # always holds one part. Joined end to end when it holds more.
        points: list[np.ndarray] = []
        for part in parts:
            array = np.asarray(part, dtype=np.float64)
            if array.ndim != 2 or len(array) < 2:
                continue
            array = array[:, :2]
            if points and np.allclose(points[-1][-1], array[0]):
                array = array[1:]
            points.append(array)
        if not points:
            continue
        line = np.vstack(points)
        # Duplicate vertices have no direction and break every tangent.
        keep = np.concatenate([[True], np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-6])
        line = line[keep]
        if len(line) < 2:
            continue
        try:
            start = int(props.get("jte_id_beg"))
            end = int(props.get("jte_id_end"))
        except (TypeError, ValueError):
            continue
        network.lines.append(
            Centreline(
                points=line,
                section_id=int(props.get("wvk_id") or len(network.lines)),
                start_junction=start,
                end_junction=end,
                name=str(props.get("stt_naam") or "").strip(),
                kind=str(props.get("bst_code") or "").strip(),
            )
        )
    return network


def fetch_centrelines(
    bbox: BBox,
    *,
    url: str = NWB_URL,
    page_limit: int = 1000,
    timeout: float = 120.0,
    max_retries: int = 3,
    max_pages: int = 50,
    cache_dir: Path | None = None,
) -> Network:
    """Every road section touching the bbox, or an empty network on failure.

    Empty rather than raising: without centrelines the roads are still built,
    each surface as a face of its own outline, which is a worse mesh to edit
    but not a wrong one. A run should not die over it.
    """
    cache = None
    if cache_dir is not None:
        key = f"{bbox.xmin:.1f}_{bbox.ymin:.1f}_{bbox.xmax:.1f}_{bbox.ymax:.1f}"
        cache = Path(cache_dir) / f"nwb_wegvakken_{key}.json.gz"
        if cache.is_file():
            with gzip.open(cache, "rt", encoding="utf-8") as handle:
                features = json.load(handle)
            network = parse_features(features)
            LOG.info("NWB: %d road sections (cached)", len(network))
            return network

    params: dict | None = {
        "bbox": f"{bbox.xmin:.3f},{bbox.ymin:.3f},{bbox.xmax:.3f},{bbox.ymax:.3f}",
        "bbox-crs": RD_URI,
        "crs": RD_URI,
        "limit": page_limit,
        "f": "json",
    }
    features: list[dict] = []
    next_url: str | None = url
    seen: set[str] = set()
    try:
        for page in range(max_pages):
            response = get_with_retry(
                next_url,
                params=params,
                timeout=timeout,
                max_retries=max_retries,
                description=f"NWB wegvakken page {page}",
            )
            payload = response.json()
            features.extend(payload.get("features") or [])
            links = [
                link.get("href")
                for link in payload.get("links") or []
                if link.get("rel") == "next" and link.get("href")
            ]
            if not links or links[0] in seen:
                break
            seen.add(links[0])
            next_url, params = links[0], None
    except Exception as exc:  # noqa: BLE001 - see the docstring
        LOG.warning(
            "could not read NWB road centrelines (%s); roads are built from "
            "their outlines alone, without edge loops across the street",
            short_error(exc),
        )
        return Network(source="unavailable")

    network = parse_features(features)
    LOG.info("NWB: %d road sections, %.1f km", len(network), sum(l.length for l in network.lines) / 1000)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        staging = cache.with_suffix(cache.suffix + ".part")
        with gzip.open(staging, "wt", encoding="utf-8") as handle:
            json.dump(features, handle)
        staging.replace(cache)
    return network


__all__ = ["Centreline", "Network", "NWB_URL", "fetch_centrelines", "parse_features"]
