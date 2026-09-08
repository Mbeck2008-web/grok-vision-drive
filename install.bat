@echo off
setlocal EnableExtensions EnableDelayedExpansion
title GVD Installer
cd /d "%~dp0"

set "LOG=%TEMP%\gvd_install.log"
echo. > "%LOG%"

set "USER_ROOT="
if exist "%LOCALAPPDATA%\BeamNG.drive\" set "USER_ROOT=%LOCALAPPDATA%\BeamNG.drive"
if not defined USER_ROOT if exist "%LOCALAPPDATA%\BeamNG\BeamNG.drive\" set "USER_ROOT=%LOCALAPPDATA%\BeamNG\BeamNG.drive"

if not defined USER_ROOT (
  echo [GVD] Could not find BeamNG user folder under LOCALAPPDATA.
  echo Tried: %%LOCALAPPDATA%%\BeamNG.drive and %%LOCALAPPDATA%%\BeamNG\BeamNG.drive\
  pause
  exit /b 1
)

echo [GVD] User root: %USER_ROOT%
echo [GVD] Candidate version folders with mods:
for /d %%D in ("%USER_ROOT%\*") do (
  if exist "%%D\mods\" echo   %%D
)

rem Numeric / current pick: current wins; else highest 0.xx by real version (0.32 > 0.9)
set "BEST_PATH="
set "BEST_VER="
for /f "usebackq delims=" %%L in (`powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=$env:USER_ROOT; Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'mods') } | ForEach-Object { $n=$_.Name; if ($n -ieq 'current') { [pscustomobject]@{ Name=$n; Path=$_.FullName; Key=[version]'9999.0' } } elseif ($n -match '^(\d+)\.(\d+)') { [pscustomobject]@{ Name=$n; Path=$_.FullName; Key=[version]($Matches[1]+'.'+$Matches[2]) } } else { [pscustomobject]@{ Name=$n; Path=$_.FullName; Key=[version]'0.0' } } } | Sort-Object Key -Descending | Select-Object -First 1 | ForEach-Object { $_.Path + '|' + $_.Name }"`) do (
  for /f "tokens=1,2 delims=|" %%A in ("%%L") do (
    set "BEST_PATH=%%A"
    set "BEST_VER=%%B"
  )
)

if not defined BEST_PATH (
  echo [GVD] No version folder with a mods directory under %USER_ROOT%
  pause
  exit /b 1
)

echo [GVD] Installing into newest: %BEST_PATH%  (version=%BEST_VER%)

set "MODS=%BEST_PATH%\mods"
set "DEST=%MODS%\unpacked\gvd"
set "ZIP=%MODS%\gvd.zip"

if not exist "%~dp0beamng_mod\" (
  echo [GVD] Missing beamng_mod\ next to install.bat
  pause
  exit /b 1
)

if exist "%DEST%\" rd /s /q "%DEST%"
mkdir "%DEST%" 2>nul

echo [GVD] Copying beamng_mod\ -^> %DEST%
xcopy /E /I /Y "%~dp0beamng_mod\*" "%DEST%\" >nul
if errorlevel 1 (
  echo [GVD] Copy failed.
  pause
  exit /b 1
)

where tar >nul 2>&1
if %ERRORLEVEL%==0 (
  if exist "%ZIP%" del /f /q "%ZIP%"
  pushd "%DEST%"
  tar -a -cf "%ZIP%" * >nul 2>&1
  popd
  if exist "%ZIP%" echo [GVD] Also wrote %ZIP%
) else (
  echo [GVD] tar not found - skipped gvd.zip (unpacked install is enough)
)

set "DOCS=%USERPROFILE%\Documents\GVD"
if not exist "%DOCS%\" mkdir "%DOCS%"
if exist "%~dp0python\" xcopy /E /I /Y "%~dp0python\*" "%DOCS%\python\" >nul
if exist "%~dp0config\" xcopy /E /I /Y "%~dp0config\*" "%DOCS%\config\" >nul

echo installed=%DEST% version=%BEST_VER% > "%LOG%"
echo [GVD] Log: installed=%DEST% version=%BEST_VER%
type "%LOG%"

echo.
echo Installed. Enable GVD in Mod Manager if it is off. Press any key.
pause >nul
endlocal
exit /b 0
