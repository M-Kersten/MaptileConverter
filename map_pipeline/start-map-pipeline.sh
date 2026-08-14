#!/bin/bash
# Double-click this to start the map pipeline. No terminal knowledge needed.
#
# First run sets up a private Python environment next to this file and
# downloads about a gigabyte of dependencies, most of which is Blender.
# Later runs skip straight to opening the page.
cd "$(dirname "$0")" || exit 1

VENV=".venv"
STAMP="$VENV/.installed"

# bpy - Blender as a library - publishes wheels for CPython 3.11 only. On any
# other version pip quietly installs everything else and the run then fails
# several minutes in, at the Blender stage, with nothing obviously wrong. So
# the version is checked here rather than discovered there.
PY=""
for candidate in python3.11 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)' 2>/dev/null; then
    PY="$candidate"
    break
  fi
done

if [ -z "$PY" ]; then
  cat <<'MSG'

  Python 3.11 was not found, and this needs that exact version:
  Blender only publishes its Python library for 3.11.

  On Debian or Ubuntu:

      sudo apt install python3.11 python3.11-venv

  or download it from
  https://www.python.org/downloads/release/python-3119/

MSG
  read -r -p "Press return to close."
  exit 1
fi

if [ ! -d "$VENV" ]; then
  echo "Creating a private Python environment. This happens once."
  "$PY" -m venv "$VENV" || { read -r -p "Setup failed. Press return."; exit 1; }
fi

if [ ! -f "$STAMP" ]; then
  echo
  echo "Installing dependencies. This is about a gigabyte and takes a while."
  echo "You only pay for this once."
  echo
  "$VENV/bin/python" -m pip install --upgrade pip || { read -r -p "Setup failed. Press return."; exit 1; }
  "$VENV/bin/python" -m pip install -r requirements.txt || { read -r -p "Setup failed. Press return."; exit 1; }
  echo installed > "$STAMP"
fi

echo
echo "Starting. Your browser should open at http://127.0.0.1:8765"
echo "Close this window when you are finished."
echo
exec "$VENV/bin/python" ui/server.py --open
