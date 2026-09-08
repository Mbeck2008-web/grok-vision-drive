@echo off
setlocal EnableExtensions EnableDelayedExpansion
title GVD Uninstaller
cd /d "%~dp0"

set "USER_ROOT="
if exist "%LOCALAPPDATA%\BeamNG.drive\" set "USER_ROOT=%LOCALAPPDATA%\BeamNG.drive"
if not defined USER_ROOT if exist "%LOCALAPPDATA%\BeamNG\BeamNG.drive\" set "USER_ROOT=%LOCALAPPDATA%\BeamNG\BeamNG.drive"

if not defined USER_ROOT (
  echo [GVD] Could not find BeamNG user folder.
  pause
  exit /b 1
)

set "REMOVED=0"
for /d %%D in ("%USER_ROOT%\*") do (
  if exist "%%D\mods\unpacked\gvd\" (
    echo [GVD] Removing %%D\mods\unpacked\gvd
    rd /s /q "%%D\mods\unpacked\gvd"
    set "REMOVED=1"
  )
  if exist "%%D\mods\gvd.zip" (
    echo [GVD] Removing %%D\mods\gvd.zip
    del /f /q "%%D\mods\gvd.zip"
    set "REMOVED=1"
  )
)

if "!REMOVED!"=="0" (
  echo [GVD] Nothing to remove (no unpacked\gvd or gvd.zip found).
) else (
  echo [GVD] Uninstall done. Other mods untouched.
)
pause
endlocal
exit /b 0
