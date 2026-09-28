cmake_minimum_required(VERSION 3.15...4.2)

if(WIN32)
  if(NOT DEFINED CMAKE_GENERATOR_PLATFORM)
    message(FATAL_ERROR "CMAKE_GENERATOR_PLATFORM is not defined; configure with -A x64")
  elseif(NOT DEFINED ENV{VCPKG_ROOT})
    message(FATAL_ERROR "VCPKG_ROOT is not defined")
  endif()

  set(VCPKG_TARGET_TRIPLET "x64-windows")
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE FILEPATH "vcpkg toolchain")
endif()

project(gfootball_training LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
  add_compile_options(/utf-8)
endif()

add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../engine" engine EXCLUDE_FROM_ALL)

if(WIN32)
  find_program(GFOOTBALL_HOST_PYTHON NAMES python
    NO_CMAKE_PATH
    NO_CMAKE_ENVIRONMENT_PATH
    REQUIRED)
  execute_process(
    COMMAND "${GFOOTBALL_HOST_PYTHON}" -c
            "import sysconfig; print(sysconfig.get_paths()['include']); print(sysconfig.get_config_var('LIBDIR')); print(sysconfig.get_config_var('LDLIBRARY'))"
    OUTPUT_VARIABLE GFOOTBALL_PYTHON_CONFIG
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  string(REPLACE "\r\n" ";" GFOOTBALL_PYTHON_CONFIG "${GFOOTBALL_PYTHON_CONFIG}")
  string(REPLACE "\n" ";" GFOOTBALL_PYTHON_CONFIG "${GFOOTBALL_PYTHON_CONFIG}")
  list(GET GFOOTBALL_PYTHON_CONFIG 0 GFOOTBALL_PYTHON_INCLUDE)
  list(GET GFOOTBALL_PYTHON_CONFIG 1 GFOOTBALL_PYTHON_LIBRARY_DIRECTORY)
  list(GET GFOOTBALL_PYTHON_CONFIG 2 GFOOTBALL_PYTHON_LIBRARY_NAME)
  string(REPLACE ".dll" ".lib" GFOOTBALL_PYTHON_LIBRARY_NAME
                 "${GFOOTBALL_PYTHON_LIBRARY_NAME}")
  find_library(GFOOTBALL_PYTHON_LIBRARY
    NAMES "${GFOOTBALL_PYTHON_LIBRARY_NAME}"
    PATHS "${GFOOTBALL_PYTHON_LIBRARY_DIRECTORY}"
    NO_DEFAULT_PATH REQUIRED)
  add_library(_gfootball_env MODULE
    binding.cpp
    environment.cpp
    environment.hpp)
  target_include_directories(_gfootball_env SYSTEM PRIVATE
    "${GFOOTBALL_PYTHON_INCLUDE}")
  target_link_libraries(_gfootball_env PRIVATE "${GFOOTBALL_PYTHON_LIBRARY}")
  set_target_properties(_gfootball_env PROPERTIES PREFIX "" SUFFIX ".pyd")
else()
  find_package(Python3 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
  Python3_add_library(_gfootball_env MODULE
    binding.cpp
    environment.cpp
    environment.hpp
    WITH_SOABI)
endif()

target_link_libraries(_gfootball_env PRIVATE gfootball_engine)
target_include_directories(_gfootball_env PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../engine/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/../../engine/src/cmake")

set_target_properties(_gfootball_env PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
foreach(configuration DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
  set_target_properties(_gfootball_env PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY_${configuration} "${CMAKE_CURRENT_SOURCE_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_${configuration} "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
