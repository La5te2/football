@echo off
setlocal

set "JACKAROO_DIR=%~dp0"
set "ROOT=%JACKAROO_DIR%..\.."
if "%VCPKG_ROOT%"=="" (
  echo VCPKG_ROOT is not set.
  echo Example: set "VCPKG_ROOT=^<path-to-vcpkg^>"
  exit /b 1
)
if not exist "%VCPKG_ROOT%\vcpkg.exe" (
  echo vcpkg.exe was not found under VCPKG_ROOT: %VCPKG_ROOT%
  exit /b 1
)

if "%BUILD_CONFIGURATION%"=="" set "BUILD_CONFIGURATION=Release"
set "BUILD_DIR=%ROOT%\.local\build\jackaroo-windows"

cmake -S "%JACKAROO_DIR%." -B "%BUILD_DIR%" -A x64 ^
  -DVCPKG_MANIFEST_MODE=OFF
if errorlevel 1 exit /b 1

cmake --build "%BUILD_DIR%" --parallel --config %BUILD_CONFIGURATION% ^
  --target _gfootball_env
