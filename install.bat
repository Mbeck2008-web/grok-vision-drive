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
echo [GVD] Scanning for versioned folders with mods...

set "BEST_VER="
set "BEST_PATH="

for /d %%D in ("%USER_ROOT%\*") do (
  if exist "%%D\mods\" (
    set "NAME=%%~nxD"
    rem Prefer lexical newest for 0.xx / current — track best by name
    if /I "!NAME!"=="current" (
      set "BEST_VER=!NAME!"
      set "BEST_PATH=%%D"
    ) else (
      if not defined BEST_PATH (
        set "BEST_VER=!NAME!"
        set "BEST_PATH=%%D"
      ) else (
        if /I not "!BEST_VER!"=="current" (
          if "!NAME!" GTR "!BEST_VER!" (
            set "BEST_VER=!NAME!"
            set "BEST_PATH=%%D"
          )
        )
      )
    )
  )
)

if not defined BEST_PATH (
  echo [GVD] No version folder with a mods directory under %USER_ROOT%
  pause
  exit /b 1
)

echo [GVD] Candidate version folders with mods:
for /d %%D in ("%USER_ROOT%\*") do (
  if exist "%%D\mods\" echo   %%D
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

rem Optional zip of unpacked tree (unpacked is enough)
where tar >nul 2>&1
if %ERRORLEVEL%==0 (
  if exist "%ZIP%" del /f /q "%ZIP%"
  pushd "%DEST%"
  tar -a -cf "%ZIP%" * >nul 2>&1
  popd
  if exist "%ZIP%" echo [GVD] Also wrote %ZIP%
) else (
  echo [GVD] tar not found — skipped gvd.zip (unpacked install is enough)
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
