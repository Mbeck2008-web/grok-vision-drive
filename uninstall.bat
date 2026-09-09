@echo off
setlocal EnableExtensions EnableDelayedExpansion
title GVD Uninstaller
cd /d "%~dp0"

set "ROOT_A=%LOCALAPPDATA%\BeamNG\BeamNG.drive"
set "ROOT_B=%LOCALAPPDATA%\BeamNG.drive"

echo [GVD] Scanning for gvd under:
echo   %ROOT_A%
echo   %ROOT_B%

set "REMOVED=0"

rem Prefer listing current first
if exist "%ROOT_A%\current\mods\unpacked\gvd\" (
  echo [GVD] Candidate: %ROOT_A%\current\mods\unpacked\gvd
)
if exist "%ROOT_A%\current\mods\gvd.zip" (
  echo [GVD] Candidate: %ROOT_A%\current\mods\gvd.zip
)

if exist "%ROOT_A%\" for /d %%D in ("%ROOT_A%\*") do (
  if exist "%%D\mods\unpacked\gvd\" echo [GVD] Candidate: %%D\mods\unpacked\gvd
  if exist "%%D\mods\gvd.zip" echo [GVD] Candidate: %%D\mods\gvd.zip
)
if exist "%ROOT_B%\" for /d %%D in ("%ROOT_B%\*") do (
  if exist "%%D\mods\unpacked\gvd\" echo [GVD] Candidate: %%D\mods\unpacked\gvd
  if exist "%%D\mods\gvd.zip" echo [GVD] Candidate: %%D\mods\gvd.zip
)

rem Remove from both trees (safe — only gvd paths)
for %%R in ("%ROOT_A%" "%ROOT_B%") do (
  if exist %%~R\ (
    for /d %%D in ("%%~R\*") do (
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
