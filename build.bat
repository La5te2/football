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

if exist "%RUNTIME_DIR%" cmake -E remove_directory "%RUNTIME_DIR%"
if errorlevel 1 exit /b 1
mkdir "%RUNTIME_DIR%"
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\runtime\gfootball.exe" "%RUNTIME_DIR%\" >nul
if errorlevel 1 exit /b 1
copy /Y "%BUILD_DIR%\runtime\replay.exe" "%RUNTIME_DIR%\" >nul
if errorlevel 1 exit /b 1
for %%F in ("%BUILD_DIR%\runtime\*.dll") do (
  copy /Y "%%~fF" "%RUNTIME_DIR%\" >nul
  if errorlevel 1 exit /b 1
)
if exist "%BUILD_DIR%\runtime\models" (
  xcopy /E /I /Y "%BUILD_DIR%\runtime\models" "%RUNTIME_DIR%\models" >nul
  if errorlevel 1 exit /b 1
)
xcopy /E /I /Y "%ROOT%engine\data" "%RUNTIME_DIR%\data" >nul
if errorlevel 1 exit /b 1
xcopy /E /I /Y "%ROOT%engine\fonts" "%RUNTIME_DIR%\fonts" >nul
if errorlevel 1 exit /b 1

echo Native runtime created in %RUNTIME_DIR%
