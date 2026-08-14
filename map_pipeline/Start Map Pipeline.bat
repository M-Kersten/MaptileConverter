@echo off
rem Double-click this to start the map pipeline. No terminal knowledge needed.
rem
rem First run sets up a private Python environment next to this file and
rem downloads about a gigabyte of dependencies, most of which is Blender.
rem Later runs skip straight to opening the page.
setlocal
cd /d "%~dp0"
title Map pipeline

set "VENV=.venv"
set "STAMP=%VENV%\.installed"

rem bpy - Blender as a library - publishes wheels for CPython 3.11 only. On any
rem other version pip quietly installs everything else and the run then fails
rem several minutes in, at the Blender stage, with nothing obviously wrong.
rem So the version is checked here rather than discovered there.
set "PY="
for %%V in (3.11) do (
  py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
)
if not defined PY (
  python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
  echo.
  echo   Python 3.11 was not found, and this needs that exact version:
  echo   Blender only publishes its Python library for 3.11.
  echo.
  echo   Install it from https://www.python.org/downloads/release/python-3119/
  echo   and tick "Add python.exe to PATH" in the installer.
  echo.
  echo   Opening the download page...
  start "" "https://www.python.org/downloads/release/python-3119/"
  echo.
  pause
  exit /b 1
)

if not exist "%VENV%" (
  echo Creating a private Python environment. This happens once.
  %PY% -m venv "%VENV%" || goto :failed
)

if not exist "%STAMP%" (
  echo.
  echo Installing dependencies. This is about a gigabyte and takes a while.
  echo You only pay for this once.
  echo.
  "%VENV%\Scripts\python.exe" -m pip install --upgrade pip || goto :failed
  "%VENV%\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed
  echo installed > "%STAMP%"
)

echo.
echo Starting. Your browser should open at http://127.0.0.1:8765
echo Close this window when you are finished.
echo.
"%VENV%\Scripts\python.exe" ui\server.py --open
goto :eof

:failed
echo.
echo   Setup failed. The messages above say why; the usual causes are no
echo   internet connection or a proxy blocking pypi.org.
echo.
pause
exit /b 1
