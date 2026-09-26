@echo off
setlocal

set "ROOT=%~dp0"
if "%VCPKG_ROOT%"=="" (
  echo VCPKG_ROOT is not set.
  echo Example: set "VCPKG_ROOT=^<path-to-vcpkg^>"
  exit /b 1
)
if not exist "%VCPKG_ROOT%\vcpkg.exe" (
  echo vcpkg.exe was not found under VCPKG_ROOT: %VCPKG_ROOT%
  exit /b 1
)

if "%GENERATOR_PLATFORM%"=="" set "GENERATOR_PLATFORM=x64"
if "%BUILD_CONFIGURATION%"=="" set "BUILD_CONFIGURATION=Release"
set "BUILD_DIR=%ROOT%.local\build\native-windows"
set "RUNTIME_DIR=%ROOT%bin"

cmake -S "%ROOT%." -B "%BUILD_DIR%" -A %GENERATOR_PLATFORM% ^
  -DVCPKG_MANIFEST_MODE=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --parallel --config %BUILD_CONFIGURATION%
if errorlevel 1 exit /b 1

if not exist "%RUNTIME_DIR%" mkdir "%RUNTIME_DIR%"
copy /Y "%BUILD_DIR%\runtime\gfootball.exe" "%RUNTIME_DIR%\" >nul
copy /Y "%BUILD_DIR%\runtime\replay.exe" "%RUNTIME_DIR%\" >nul
for %%F in ("%BUILD_DIR%\runtime\*.dll") do (
  if /I not "%%~nxF"=="tamakeri.dll" copy /Y "%%~fF" "%RUNTIME_DIR%\" >nul
)
xcopy /E /I /Y "%ROOT%engine\data" "%RUNTIME_DIR%\data" >nul
xcopy /E /I /Y "%ROOT%engine\fonts" "%RUNTIME_DIR%\fonts" >nul

echo Native runtime created in %RUNTIME_DIR%
