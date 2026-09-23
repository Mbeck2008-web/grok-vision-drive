@echo off
setlocal EnableExtensions
title GVD Tech
cd /d "%~dp0"

rem Tech launcher: 8-cam BeamNGpy + vehicle electrics/pose/damage + vehicle.control.
rem One starter, already up: install-root BeamNG.tech.exe -tcom -console -gfx dx11.
rem Does NOT launch Steam BeamNG.drive. Does NOT start BeamNG.tech. Extra args pass through to run_vision.py.
rem GVD_TECH_LAUNCH=0 attaches. Never also launch=1. tech.key and user path do not block the wait gate.

set "GVD_BACKEND=beamngpy"
set "GVD_BEAMNG=1"
set "GVD_PRODUCT=tech"
set "GVD_TECH_LAUNCH=0"
set "GVD_DOCS_DIR=%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\Documents\GVD"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"

set "RUN="
set "RUNROOT=%~dp0"
if exist "%~dp0python\run_vision.py" set "RUN=%~dp0python\run_vision.py"
if not defined RUN if exist "%GVD_DOCS_DIR%\python\run_vision.py" (
  set "RUN=%GVD_DOCS_DIR%\python\run_vision.py"
  set "RUNROOT=%GVD_DOCS_DIR%"
)
set "REQS=%~dp0requirements-beamng.txt"

if not defined BNG_HOME if defined BEAMNG_HOME set "BNG_HOME=%BEAMNG_HOME%"

if not defined RUN (
  echo [GVD] python\run_vision.py not found. Run install.bat first.
  pause
  exit /b 1
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
echo [GVD] bus: %GVD_DOCS_DIR%
echo [GVD] product=tech. Supervisor and GELua must print the same python_bus/lua_bus folder.
echo [GVD] link=MISMATCH means Drive vs Tech, a stale lua_bus, or leftover GVD_DOCS_DIR. Live lua_bus wins. Live Tech 8-cam UNPROVEN.
echo [GVD] Match BeamNGpy to the Tech build in use (0.38 -^> 1.35.x, 0.39 -^> 1.36). config\tech.yaml beamngpy_pin.
echo [GVD] GVD_TECH_LAUNCH=0 attach. This bat does not start BeamNG.tech.
echo [GVD] One starter only. Already running: "%BNG_HOME%\BeamNG.tech.exe" -tcom -console -gfx dx11
echo [GVD] -gfx dx11 avoids the D3D12 Basic Render crash. A bare Bin64 start is not this starter.
echo [GVD] Do not combine a bat start with GVD_TECH_LAUNCH=1. That double-starts Tech.
echo [GVD] tech.key status: non-empty "%BNG_HOME%\tech.key". Missing or empty does not block the wait gate.
echo [GVD] User path is optional. An empty user path does not block the wait gate.
echo [GVD] Mod: Tech current\mods\unpacked\gvd. Spawn a vehicle before this bat.
echo [GVD] Wait gate before vision/Hz: port 25252 LISTENING, mod, vehicle, fresh lua_bus, buses_same.
echo [GVD] Esc or q in GVD VISION disconnects only (quit_on_close=false). Do not kill BeamNG.tech or CrashSender.
echo [GVD] Preflight:
%PY% "%RUN%" --tech-hold
if errorlevel 1 (
  echo [GVD] REFUSE: Tech hold preflight failed. Supervisor not started.
  echo [GVD] Unique-frame Hz is not measured until the wait gate passes. Live Hz is not claimed here.
  pause
  exit /b 1
)
echo [GVD] Starting supervisor: %RUN%  (backend=beamngpy, attach)
start "GVD supervisor" /D "%RUNROOT%" cmd /k %PY% "%RUN%" --backend beamngpy --viz %*

echo [GVD] Mods: run install.bat so unpacked\gvd lands in Drive and (if present) Tech current\mods.
echo [GVD] Prove: Tech already up via BeamNG.tech.exe -tcom -console -gfx dx11, mod enabled, vehicle spawned, then this bat attaches.
echo [GVD] Soft Esc of the supervisor disconnects only. Killing BeamNG.tech is a failed prove: the port dies, lua_bus goes stale, unique-frame Hz stays 0.
echo [GVD] Optional pin: edit config\tech.yaml nav.pin_lat / pin_lon. Hint only — GVD is not routing to the pin yet.
echo [GVD] Probe without driving:  %PY% "%RUN%" --tech-probe
echo [GVD] Confirm both logs show the same python_bus/lua_bus folder and link=ok. Do not claim dual-monitor / Tech 8-cam / unique-frame Hz proven.
echo [GVD] Quit supervisor: q or Esc in GVD VISION (disconnect only).
pause
endlocal
exit /b 0
