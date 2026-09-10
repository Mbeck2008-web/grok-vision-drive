@echo off
setlocal EnableExtensions EnableDelayedExpansion
title GVD Installer
cd /d "%~dp0"

set "LOG=%TEMP%\gvd_install.log"
echo. > "%LOG%"

rem Roots: modern layout is LOCALAPPDATA\BeamNG\BeamNG.drive\current\mods (0.38+)
rem Legacy: LOCALAPPDATA\BeamNG.drive\<ver>\mods
set "ROOT_A=%LOCALAPPDATA%\BeamNG\BeamNG.drive"
set "ROOT_B=%LOCALAPPDATA%\BeamNG.drive"

echo [GVD] Scanning BeamNG mod roots...
echo   A: %ROOT_A%
echo   B: %ROOT_B%

rem Prefer current\mods when present (BeamNG 0.38.6 Mod Manager path)
set "BEST_PATH="
set "BEST_VER="
if exist "%ROOT_A%\current\mods\" (
  set "BEST_PATH=%ROOT_A%\current"
  set "BEST_VER=current"
  echo [GVD] Prefer: %ROOT_A%\current\mods  (version=current)
)

echo [GVD] Candidate version folders with mods:
if exist "%ROOT_A%\" for /d %%D in ("%ROOT_A%\*") do if exist "%%D\mods\" echo   %%D
if exist "%ROOT_B%\" for /d %%D in ("%ROOT_B%\*") do if exist "%%D\mods\" echo   %%D

if not defined BEST_PATH (
  rem Else newest versioned under both roots (current Key=9999 already preferred above)
  set "ROOT_A=%ROOT_A%"
  set "ROOT_B=%ROOT_B%"
  for /f "usebackq delims=" %%L in (`powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$roots=@($env:ROOT_A,$env:ROOT_B); $cands=@(); foreach($root in $roots){ if(-not (Test-Path -LiteralPath $root)){continue}; Get-ChildItem -LiteralPath $root -Directory -ErrorAction SilentlyContinue | Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName 'mods') } | ForEach-Object { $n=$_.Name; if($n -ieq 'current'){ $key=[version]'9999.0' } elseif($n -match '^(\d+)\.(\d+)'){ $key=[version]($Matches[1]+'.'+$Matches[2]) } else { $key=[version]'0.0' }; $cands += [pscustomobject]@{ Name=$n; Path=$_.FullName; Key=$key; Root=$root } } }; if(-not $cands){ '' } else { ($cands | Sort-Object Key -Descending | Select-Object -First 1 | ForEach-Object { $_.Path + '|' + $_.Name }) }"`) do (
    for /f "tokens=1,2 delims=|" %%A in ("%%L") do (
      set "BEST_PATH=%%A"
      set "BEST_VER=%%B"
    )
  )
)

if not defined BEST_PATH (
  echo [GVD] No BeamNG mods folder found.
  echo Tried: %%LOCALAPPDATA%%\BeamNG\BeamNG.drive\current\mods
  echo        %%LOCALAPPDATA%%\BeamNG\BeamNG.drive\^<ver^>\mods
  echo        %%LOCALAPPDATA%%\BeamNG.drive\^<ver^>\mods
  pause
  exit /b 1
)

echo [GVD] Chosen install: %BEST_PATH%\mods  (version=%BEST_VER%)

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
echo Installed. Fully quit BeamNG, relaunch, enable Grok Vision Drive in Mod Manager.

rem Also copy into BeamNG.tech user mods when that folder exists (separate from Drive).
set "TECH_DEST="
if exist "%LOCALAPPDATA%\BeamNG.tech\current\mods\" set "TECH_DEST=%LOCALAPPDATA%\BeamNG.tech\current\mods\unpacked\gvd"
if not defined TECH_DEST if exist "%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\mods\" set "TECH_DEST=%LOCALAPPDATA%\BeamNG\BeamNG.tech\current\mods\unpacked\gvd"
if defined TECH_DEST (
  if exist "%TECH_DEST%\" rd /s /q "%TECH_DEST%"
  mkdir "%TECH_DEST%" 2>nul
  xcopy /E /I /Y "%~dp0beamng_mod\*" "%TECH_DEST%\" >nul
  echo [GVD] Also installed Tech mods: %TECH_DEST%
) else (
  echo [GVD] No BeamNG.tech current\mods folder yet - Drive install is enough until Tech is on this box.
)

pause >nul
endlocal
exit /b 0
