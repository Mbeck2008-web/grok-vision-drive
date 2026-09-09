@echo off
setlocal EnableExtensions
title GVD Play
cd /d "%~dp0"

rem Retail launcher (M6): 1-cam window capture + GVD VISION window + Steam BeamNG.
rem Honesty: retail = ONE window capture (main cam). 8-cam rig + driving = BeamNG.tech + BeamNGpy.
rem Tech users: set GVD_BACKEND=beamngpy before running. Extra args pass through to run_vision.py.
if not defined GVD_BACKEND set "GVD_BACKEND=window"

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
set "REQS=%~dp0requirements-retail.txt"

if not defined RUN (
  echo [GVD] python\run_vision.py not found - skipped supervisor. Run install.bat first.
  goto :launch
)
if not defined PY (
  echo [GVD] Python 3 not found - skipped supervisor. Install Python 3 from python.org and tick "Add to PATH".
  goto :launch
)

rem Retail runtime check: numpy + OpenCV + PyYAML + mss (window capture). Fail fast and visibly.
%PY% -c "import numpy, cv2, yaml, mss" >nul 2>&1
if not errorlevel 1 goto :supervisor

echo [GVD] Missing Python packages for the retail runtime: numpy, opencv-python, PyYAML, mss.
if not exist "%REQS%" (
  echo [GVD] Run:  %PY% -m pip install numpy opencv-python PyYAML mss    then re-run play_gvd.bat
  goto :launch
)
choice /C YN /N /M "[GVD] Install them now with pip -r requirements-retail.txt? [Y/N] "
if errorlevel 2 (
  echo [GVD] Skipped supervisor. Later:  %PY% -m pip install -r "%REQS%"
  goto :launch
)
%PY% -m pip install -r "%REQS%"
%PY% -c "import numpy, cv2, yaml, mss" >nul 2>&1
if errorlevel 1 (
  echo [GVD] Packages still missing - skipped supervisor. See pip output above.
  goto :launch
)

:supervisor
echo [GVD] Starting supervisor: %RUN%  (backend=%GVD_BACKEND%)
echo [GVD] Retail = 1 window capture, main cam only. 8-cam rig + driving need BeamNG.tech + BeamNGpy.
rem cmd /k keeps the supervisor console open so errors stay readable.
start "GVD supervisor" /D "%RUNROOT%" cmd /k %PY% "%RUN%" --backend %GVD_BACKEND% --viz %*

:launch
echo [GVD] Trying Steam app 284160. If BeamNG does not open, launch it yourself.
start "" "steam://rungameid/284160" 2>nul
echo [GVD] Mods expect: %%LOCALAPPDATA%%\BeamNG\BeamNG.drive\current\mods\unpacked\gvd  (0.38+)
echo [GVD] Legacy also: %%LOCALAPPDATA%%\BeamNG.drive\^<ver^>\mods\unpacked\gvd
echo [GVD] Run install.bat if Mod Manager is empty. Alt+A or the GVD app = Engage (ribbon + HUD).
echo [GVD] Retail does NOT steer the car (no BeamNGpy vehicle handle). Quit supervisor: q in GVD VISION.
pause
endlocal
exit /b 0
