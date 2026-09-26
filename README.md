# A Modified Gameplay Football

This repository contains a native 11-vs-11 football simulation based on the open-source Gameplay Football engine. It provides an OpenGL match executable, built-in team AI, and eternal model plugin support.

Useful links:

* [Original Gameplay Football repository](https://github.com/BazkieBumpercar/GameplayFootball)
* [Google Football Kaggle competition](https://www.kaggle.com/c/google-football)

We thank Bastiaan Konings Schuiling, who authored and open-sourced Gameplay Football.

## Building

The Windows build requires a 64-bit Visual Studio C++ toolchain, CMake, vcpkg, and Torch. Set `VCPKG_ROOT` to the vcpkg checkout before building:

```powershell
$env:VCPKG_ROOT = '<path-to-vcpkg>'
.\build.bat
```

CMake obtains the LibTorch package from the active Python `torch` installation by default. Set `GFOOTBALL_TORCH_ROOT` when using a standalone LibTorch distribution.

On Linux, install the equivalent C++ dependencies and run:

```bash
./build.sh
```

Build intermediates remain under `.local/build/`. The runnable files and engine assets are copied to `bin/`.

## Playing a Match

Running the executable without side options starts built-in AI versus built-in AI(both `builtin`):

```powershell
.\bin\gfootball.exe
```

Use `--left` and `--right` to assign either `builtin` or a model plugin before the match starts:

```powershell
.\bin\gfootball.exe --left eternal-model.dll --right builtin
```

In a rendered match, press Space to pause or resume. Use `[` and `]` to select 0.25x, 0.5x, 1x, or 2x playback. These controls affect presentation timing only; simulation steps and model decisions retain their original cadence.

The same controller options apply to headless matches:

```powershell
.\bin\gfootball.exe --render=false --real_time=false `
  --left eternal-model.dll `
  --right eternal-model.dll
```

## License

See `LICENSE` and `engine/LICENSE`.
