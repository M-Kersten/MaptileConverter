"""The Python packages a run needs, and whether this interpreter has them.

Standard library only, because it runs before anything that imports the
packages it checks. Without it a package missing from an environment set up
before the package was added shows up minutes into a run, as a traceback from
whichever stage first reaches for it.

Keep `REQUIRED` in step with requirements.txt; a test holds the two together.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import re
import sys
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"

# (import name, pip name, lowest version that works). bpy is not here: it is
# one of three ways to reach Blender, and the Blender stage says so itself.
REQUIRED: tuple[tuple[str, str, str], ...] = (
    ("requests", "requests", "2.31"),
    ("rasterio", "rasterio", "1.3"),
    ("numpy", "numpy", "1.24"),
    ("pyproj", "pyproj", "3.5"),
    ("mapbox_earcut", "mapbox_earcut", "1.0"),
    ("PIL", "pillow", "10.0"),
    ("shapely", "shapely", "2.1"),
)


def _version(text: str) -> tuple[int, ...]:
    """The leading numbers of a version string: '2.2.0rc1' -> (2, 2, 0)."""
    parts = []
    for piece in text.split("."):
        match = re.match(r"\d+", piece)
        if match is None:
            break
        parts.append(int(match.group()))
    return tuple(parts)


def missing_dependencies() -> list[str]:
    """One line per required package that is absent or too old."""
    problems = []
    for module, package, minimum in REQUIRED:
        if importlib.util.find_spec(module) is None:
            problems.append(f"{package} is not installed")
            continue
        try:
            have = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            # Importable but not installed by pip (a system or conda
            # package without metadata): nothing to compare, so trust it.
            continue
        if _version(have) < _version(minimum):
            problems.append(f"{package} {have} is too old; {minimum} or newer is needed")
    return problems


def install_command() -> str:
    """The command that installs what is missing into this interpreter."""
    return f'"{sys.executable}" -m pip install -r "{REQUIREMENTS}"'


def report_missing(stream=None) -> bool:
    """Print what is missing and how to fix it; True when something is."""
    problems = missing_dependencies()
    if not problems:
        return False
    stream = stream or sys.stderr
    lines = [
        "This Python environment is missing packages the pipeline needs:",
        *(f"  - {problem}" for problem in problems),
        "",
        "Install them with:",
        f"  {install_command()}",
        "",
        "If you start the pipeline with the Start Map Pipeline launcher, close",
        "it and start it again: it installs whatever requirements.txt has",
        "gained since it last ran.",
    ]
    print("\n".join(lines), file=stream, flush=True)
    return True


__all__ = ["REQUIRED", "install_command", "missing_dependencies", "report_missing"]
