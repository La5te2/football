if(NOT DEFINED GFOOTBALL_TORCH_ROOT OR GFOOTBALL_TORCH_ROOT STREQUAL "")
  if(DEFINED ENV{GFOOTBALL_TORCH_ROOT})
    set(GFOOTBALL_TORCH_ROOT "$ENV{GFOOTBALL_TORCH_ROOT}")
  elseif(WIN32)
    find_program(PYTHON_LAUNCHER py REQUIRED)
    execute_process(
      COMMAND "${PYTHON_LAUNCHER}" -3.13 -c
              "import pathlib, torch; print(pathlib.Path(torch.__file__).parent)"
      OUTPUT_VARIABLE GFOOTBALL_TORCH_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
  else()
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    execute_process(
      COMMAND "${Python3_EXECUTABLE}" -c
              "import pathlib, torch; print(pathlib.Path(torch.__file__).parent)"
      OUTPUT_VARIABLE GFOOTBALL_TORCH_ROOT
      OUTPUT_STRIP_TRAILING_WHITESPACE
      COMMAND_ERROR_IS_FATAL ANY)
  endif()
endif()
file(TO_CMAKE_PATH "${GFOOTBALL_TORCH_ROOT}" GFOOTBALL_TORCH_ROOT)
find_package(Torch REQUIRED
  PATHS "${GFOOTBALL_TORCH_ROOT}/share/cmake/Torch"
  NO_DEFAULT_PATH)

add_library(tamakeri SHARED
  adapter.cpp
  adapter.hpp
  plugin.cpp
  ../../engine/src/interface.hpp)
target_include_directories(tamakeri PRIVATE . ../../engine/src)
target_include_directories(tamakeri SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
target_link_libraries(tamakeri PRIVATE ${TORCH_LIBRARIES})

if(TORCH_CXX_FLAGS)
  separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(tamakeri PRIVATE ${TORCH_CXX_FLAGS_LIST})
endif()

if(WIN32)
  target_compile_options(tamakeri PRIVATE /bigobj /wd4267)
  set_target_properties(tamakeri PROPERTIES PREFIX "")
  file(GLOB TORCH_RUNTIME_LIBRARIES
       "${GFOOTBALL_TORCH_ROOT}/lib/*.dll")
  add_custom_command(TARGET tamakeri POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:tamakeri>
            "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${TORCH_RUNTIME_LIBRARIES}
            "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMAND_EXPAND_LISTS)
endif()
