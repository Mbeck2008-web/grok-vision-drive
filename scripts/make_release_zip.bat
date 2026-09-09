@echo off
setlocal EnableExtensions
title GVD Release Zip
cd /d "%~dp0.."

rem Builds dist\gvd-retail-<version>.zip from this checkout.
rem Retail = 1-cam window capture only; no weights, no clips, no .git in the zip.

set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
  echo [GVD] Python 3 not found. Install from python.org, then re-run.
  pause
  exit /b 1
)

%PY% scripts\make_release_zip.py %*
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" echo [GVD] Release zip FAILED (rc=%RC%). See messages above.
pause
endlocal & exit /b %RC%
