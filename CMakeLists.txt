cmake_minimum_required(VERSION 3.15...4.2)

if(WIN32)
  if(NOT DEFINED CMAKE_GENERATOR_PLATFORM)
    message(FATAL_ERROR "CMAKE_GENERATOR_PLATFORM is not defined; configure with -A x64")
  elseif(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT is not defined")
  endif()

  if(CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
    set(VCPKG_TARGET_TRIPLET "x86-windows")
  elseif(CMAKE_GENERATOR_PLATFORM STREQUAL "x64")
    set(VCPKG_TARGET_TRIPLET "x64-windows")
  else()
    message(FATAL_ERROR "Only Win32 and x64 are supported")
  endif()
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE FILEPATH "vcpkg toolchain")
endif()

project(gfootball LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/utf-8)
endif()

add_subdirectory(engine)
add_subdirectory(models/tamakeri)

set_target_properties(gfootball replay PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/runtime")
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(gfootball replay PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_BINARY_DIR}/runtime")
endforeach()
