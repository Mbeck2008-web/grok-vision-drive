@echo off
setlocal EnableExtensions
title GVD Tech
cd /d "%~dp0"

rem Tech launcher: 8-cam BeamNGpy + vehicle electrics/pose/damage + vehicle.control.
rem Requires BeamNG.tech with tech.key in the install dir (not a community camera mod).
rem Does NOT launch Steam BeamNG.drive. Extra args pass through to run_vision.py.

set "GVD_BACKEND=beamngpy"
set "GVD_BEAMNG=1"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"

set "RUN="
set "RUNROOT=%~dp0"
if exist "%~dp0python\run_vision.py" set "RUN=%~dp0python\run_vision.py"
if not defined RUN if exist "%USERPROFILE%\Documents\GVD\python\run_vision.py" (
  set "RUN=%USERPROFILE%\Documents\GVD\python\run_vision.py"
  set "RUNROOT=%USERPROFILE%\Documents\GVD"
)
set "REQS=%~dp0requirements-beamng.txt"

if not defined RUN (
  echo [GVD] python\run_vision.py not found - skipped supervisor. Run install.bat first.
  goto :launch
)
if not defined PY (
  echo [GVD] Python 3 not found. Install Python 3 from python.org and tick "Add to PATH".
  pause
  exit /b 1
)

%PY% -c "import numpy, cv2, yaml, beamngpy" >nul 2>&1
if not errorlevel 1 goto :supervisor

echo [GVD] Missing Python packages for Tech: numpy, opencv-python, PyYAML, beamngpy.
if not exist "%REQS%" (
  echo [GVD] Run:  %PY% -m pip install -r requirements-beamng.txt
  pause
  exit /b 1
)
choice /C YN /N /M "[GVD] Install them now with pip -r requirements-beamng.txt? [Y/N] "
if errorlevel 2 (
  echo [GVD] Skipped. Later:  %PY% -m pip install -r "%REQS%"
  pause
  exit /b 1
)
%PY% -m pip install -r "%REQS%"
%PY% -c "import numpy, cv2, yaml, beamngpy" >nul 2>&1
if errorlevel 1 (
  echo [GVD] Packages still missing - see pip output above.
  pause
  exit /b 1
)

:supervisor
echo [GVD] Tech path: 8 RGB cameras + electrics/damage/pose + GPS nav hint (not localization, pin is not a route). LiDAR/radar optional in config\sensors.yaml — Foxglove / future fusion; planner stays vision-only.
echo [GVD] Starting supervisor: %RUN%  (backend=beamngpy)
echo [GVD] Match BeamNGpy to your Tech build (0.38 -^> 1.35.x, 0.39 -^> 1.36). Edit config\tech.yaml.
start "GVD supervisor" /D "%RUNROOT%" cmd /k %PY% "%RUN%" --backend beamngpy --viz %*

:launch
if not defined BNG_HOME if defined BEAMNG_HOME set "BNG_HOME=%BEAMNG_HOME%"
if defined BNG_HOME (
  if exist "%BNG_HOME%\BeamNG.tech.exe" (
    echo [GVD] Launching BeamNG.tech from BNG_HOME
    start "" "%BNG_HOME%\BeamNG.tech.exe"
    goto :done
  )
  if exist "%BNG_HOME%\Bin64\BeamNG.tech.x64.exe" (
    echo [GVD] Launching BeamNG.tech.x64 from BNG_HOME\Bin64
    start "" "%BNG_HOME%\Bin64\BeamNG.tech.x64.exe"
    goto :done
  )
  echo [GVD] BNG_HOME is set but no Tech exe found there: %BNG_HOME%
)
echo [GVD] Start BeamNG.tech yourself if it is not already running.
echo [GVD] Place tech.key next to the Tech exe (install dir, not the user folder).
echo [GVD] Spawn a vehicle, then the supervisor attaches cameras + electrics + GPS. Alt+G = Engage.
echo [GVD] Optional pin: edit config\tech.yaml nav.pin_lat / pin_lon. Hint only — GVD is not routing to the pin yet.
echo [GVD] Probe without driving:  %PY% "%RUN%" --tech-probe

:done
echo [GVD] Mods: run install.bat so unpacked\gvd lands in Drive and (if present) Tech current\mods.
echo [GVD] Quit supervisor: q in GVD VISION.
pause
endlocal
exit /b 0
