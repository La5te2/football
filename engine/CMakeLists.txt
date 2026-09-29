if(UNIX)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
      "${CMAKE_CURRENT_SOURCE_DIR}/CMakeModules")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS} -fPIC -g -O3")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -O3 -g")
  set(OpenGL_GL_PREFERENCE GLVND)
endif(UNIX)

# Find required libraries
FIND_PACKAGE(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

FIND_PACKAGE(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})

if(UNIX AND NOT APPLE)
  FIND_PACKAGE(EGL REQUIRED)
  include_directories(${EGL_INCLUDE_DIR})
endif()

if(WIN32)
  FIND_PACKAGE(SDL2_image CONFIG REQUIRED)
  FIND_PACKAGE(SDL2_ttf CONFIG REQUIRED)
  FIND_PACKAGE(sdl2-gfx CONFIG REQUIRED)
else()  # Unix
  FIND_PACKAGE(SDL2_image REQUIRED)
  FIND_PACKAGE(SDL2_ttf REQUIRED)
  FIND_PACKAGE(SDL2_gfx REQUIRED)
endif()

include_directories(${SDL2_IMAGE_DIRS})
include_directories(${SDL2_TTF_DIRS})
include_directories(${SDL2_GFX_DIRS})

FIND_PACKAGE(Boost REQUIRED COMPONENTS thread system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src/cmake)

# Include the sources
include(sources.cmake)

add_library(baselib OBJECT ${BASE_SOURCES} ${BASE_HEADERS}
   ${BASE_GEOMETRY_HEADERS} ${BASE_MATH_HEADERS})
add_library(systemsgraphicslib OBJECT ${SYSTEMS_GRAPHICS_SOURCES} ${SYSTEMS_COMMON_HEADERS}
   ${SYSTEMS_GRAPHICS_HEADERS} ${SYSTEMS_GRAPHICS_OBJECTS_HEADERS}
   ${SYSTEMS_GRAPHICS_RESOURCES_HEADERS} ${SYSTEMS_GRAPHICS_RENDERING_HEADERS})
add_library(loaderslib OBJECT ${LOADERS_SOURCES} ${LOADERS_HEADERS} ${MANAGERS_HEADERS})
add_library(typeslib OBJECT ${TYPES_SOURCES} ${TYPES_HEADERS})
add_library(scenelib OBJECT ${SCENE_SOURCES} ${SCENE_HEADERS}
   ${SCENE2D_HEADERS} ${SCENE3D_HEADERS} ${SCENE_OBJECTS_HEADERS}
   ${SCENE_RESOURCES_HEADERS})
add_library(utilslib OBJECT ${UTILS_SOURCES} ${UTILS_HEADERS}
   ${UTILS_EXT_HEADERS})
add_library(gui2lib OBJECT ${UTILS_GUI2_SOURCES} ${UTILS_GUI2_HEADERS}
   ${UTILS_GUI2_WIDGETS_HEADERS})

# Join all created static libraries into a single static or shared one.
set(OWN_LIBRARIES $<TARGET_OBJECTS:baselib> # $<TARGET_OBJECTS:systemscommonlib>
   $<TARGET_OBJECTS:systemsgraphicslib> $<TARGET_OBJECTS:loaderslib>
   $<TARGET_OBJECTS:typeslib> $<TARGET_OBJECTS:scenelib>
   $<TARGET_OBJECTS:utilslib> $<TARGET_OBJECTS:gui2lib>)

# Compile it as multiple static libraries
add_library(blunted2 ${BLUNTED_CORE_SOURCES} ${BLUNTED_CORE_HEADERS} ${OWN_LIBRARIES})
add_library(gamelib ${GAME_SOURCES} ${GAME_HEADERS})
add_library(menulib ${MENU_SOURCES} ${MENU_HEADERS})
add_library(datalib ${DATA_SOURCES} ${DATA_HEADERS})
add_library(enginecore STATIC
  ${CORE_SOURCES}
  ${CORE_HEADERS}
  ${ENGINE_SOURCES}
  ${ENGINE_HEADERS})

if(UNIX)
  set(SDL2_LIBS_ALL ${SDL2_IMAGE_LIBRARIES} ${SDL2_TTF_LIBRARIES} ${SDL2_GFX_LIBRARIES} ${SDL2_LIBRARIES})
elseif(WIN32)
  set(SDL2_LIBS_ALL SDL2::SDL2 SDL2_image::SDL2_image
      SDL2_ttf::SDL2_ttf SDL2::SDL2_gfx SDL2::SDL2main)
endif()

set(ENGINE_INTERNAL_LIBRARIES gamelib menulib datalib blunted2)
set(ENGINE_EXTERNAL_LIBRARIES
   Boost::filesystem Boost::system Boost::thread
   ${SDL2_LIBS_ALL} ${EGL_LIBRARIES} ${OPENGL_LIBRARIES})

add_library(gfootball_engine INTERFACE)
target_include_directories(gfootball_engine INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/src)
if(UNIX AND NOT APPLE)
  # GNU-style linkers scan static archives from left to right.  The engine's
  # internal archives reference each other, so rescan them as one group.
  target_link_libraries(gfootball_engine INTERFACE
    "-Wl,--start-group" enginecore ${ENGINE_INTERNAL_LIBRARIES}
    "-Wl,--end-group" ${ENGINE_EXTERNAL_LIBRARIES})
else()
  target_link_libraries(gfootball_engine INTERFACE
    enginecore ${ENGINE_INTERNAL_LIBRARIES} ${ENGINE_EXTERNAL_LIBRARIES})
endif()

add_executable(gfootball src/gfootball.cpp)
target_link_libraries(gfootball PRIVATE gfootball_engine)
add_executable(replay src/replaymain.cpp)
target_link_libraries(replay PRIVATE gfootball_engine)
if(WIN32)
  set_target_properties(gfootball replay PROPERTIES WIN32_EXECUTABLE TRUE)
endif()
