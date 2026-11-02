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
rem So the version is checked here rather than discovered there, and when it
rem is missing this offers to install it.
call :find_python
if defined PY goto :have_python

echo.
echo   This needs Python 3.11, and it is not on this computer. Blender only
echo   publishes its Python library for that exact version, so a newer Python
echo   you may already have cannot stand in for it.
echo.
echo   It can be installed now: the official installer from python.org, about
echo   25 MB, for your Windows user only. No administrator rights are needed,
echo   and any other Python on this computer is left as it is.
echo.
choice /c YN /m "  Install Python 3.11 now"
if errorlevel 2 goto :python_by_hand
call :install_python
call :find_python
if defined PY goto :have_python

:python_by_hand
echo.
echo   Python 3.11 is still not available. Install it yourself from
echo   https://www.python.org/downloads/release/python-3119/
echo   - the "Windows installer (64-bit)" - then start this again.
echo.
echo   Opening the download page...
start "" "https://www.python.org/downloads/release/python-3119/"
echo.
pause
exit /b 1

:have_python
if not exist "%VENV%" (
  echo Creating a private Python environment. This happens once.
  %PY% -m venv "%VENV%" || goto :failed
)

rem The stamp is a copy of the requirements.txt last installed. When an update
rem adds a package the two differ and pip runs again; it only fetches what is
rem new, so after the first time this takes seconds.
fc /b requirements.txt "%STAMP%" >nul 2>&1
if errorlevel 1 (
  echo.
  echo Installing dependencies. The first time this is about a gigabyte and
  echo takes a while; after an update it only fetches what is new.
  echo.
  "%VENV%\Scripts\python.exe" -m pip install --upgrade pip || goto :failed
  "%VENV%\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed
  copy /y requirements.txt "%STAMP%" >nul
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


rem Sets PY to a Python 3.11 command, or leaves it empty. The py launcher and
rem PATH first, then the two places the python.org installer puts it when
rem nothing was added to PATH: per user (what :install_python does) and for
rem all users.
:find_python
set "PY="
py -3.11 -c "import sys" >nul 2>&1
if not errorlevel 1 (
  set "PY=py -3.11"
  exit /b 0
)
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,11) else 1)" >nul 2>&1
if not errorlevel 1 (
  set "PY=python"
  exit /b 0
)
for %%P in ("%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%ProgramFiles%\Python311\python.exe") do (
  if exist "%%~P" (
    set PY="%%~P"
    exit /b 0
  )
)
exit /b 0

rem Downloads the python.org installer, checks it is the published file, and
rem installs it for this user without touching PATH or other Pythons.
rem 3.11.9 is the last 3.11 release python.org built Windows installers for.
:install_python
set "PY_VERSION=3.11.9"
rem SHA-256 of python-3.11.9-amd64.exe. Its MD5 is the one python.org lists
rem on the release page: e8dcd502e34932eebcaf1be056d5cbcd.
set "PY_SHA256=5ee42c4eee1e6b4464bb23722f90b45303f79442df63083f05322f1785f5fdde"
set "PY_URL=https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.exe"
set "PY_SETUP=%TEMP%\python-%PY_VERSION%-amd64.exe"
echo.
echo Downloading Python %PY_VERSION% from python.org...
curl.exe --version >nul 2>&1
if errorlevel 1 goto :download_with_powershell
curl.exe -L --fail --progress-bar -o "%PY_SETUP%" "%PY_URL%"
if errorlevel 1 goto :download_failed
goto :check_download
:download_with_powershell
powershell -NoProfile -Command "$ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -UseBasicParsing -Uri $env:PY_URL -OutFile $env:PY_SETUP"
if errorlevel 1 goto :download_failed
:check_download
powershell -NoProfile -Command "if ((Get-FileHash -Algorithm SHA256 -LiteralPath $env:PY_SETUP).Hash -ne $env:PY_SHA256) { exit 1 }"
if errorlevel 1 (
  echo.
  echo   The download does not match the file python.org publishes, so it is
  echo   not being installed.
  del "%PY_SETUP%" >nul 2>&1
  exit /b 1
)
echo Installing Python %PY_VERSION% for this user. A progress window opens for a minute.
start "" /wait "%PY_SETUP%" /passive InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Shortcuts=0
del "%PY_SETUP%" >nul 2>&1
exit /b 0
:download_failed
echo.
echo   The download failed. Check the internet connection, or whether a proxy
echo   blocks python.org.
del "%PY_SETUP%" >nul 2>&1
exit /b 1
