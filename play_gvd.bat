@echo off
setlocal EnableExtensions
title GVD Play
cd /d "%~dp0"

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"

set "RUN="
if exist "%~dp0python\run_vision.py" set "RUN=%~dp0python\run_vision.py"
if not defined RUN if exist "%USERPROFILE%\Documents\GVD\python\run_vision.py" set "RUN=%USERPROFILE%\Documents\GVD\python\run_vision.py"

if defined RUN (
  if defined PY (
    echo [GVD] Starting supervisor: %RUN%
    start "GVD supervisor" %PY% "%RUN%"
  ) else (
    echo [GVD] Python not found - skipped supervisor.
  )
) else (
  echo [GVD] python\run_vision.py not found - skipped supervisor.
)

echo [GVD] Trying Steam app 284160. If BeamNG does not open, launch it yourself.
start "" "steam://rungameid/284160" 2>nul
echo [GVD] Enable GVD in Mod Manager if needed. Alt+A toggles engage (stub).
pause
endlocal
exit /b 0
