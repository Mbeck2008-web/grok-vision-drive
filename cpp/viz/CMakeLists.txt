cmake_minimum_required(VERSION 3.20)
project(gvd_viz LANGUAGES C CXX)

# GVD C++ visualizer (docs/specs/viz_cpp.md). Reads perception-only scene
# messages; never touches the control loop.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(UNIX AND NOT APPLE)
  set(_egl_default ON)
else()
  set(_egl_default OFF)
endif()
if(WIN32)
  set(_glfw_default ON)
else()
  find_package(X11 QUIET)
  if(X11_FOUND AND X11_Xrandr_FOUND AND X11_Xinerama_FOUND AND X11_Xcursor_FOUND AND X11_Xi_FOUND)
    set(_glfw_default ON)
  else()
    set(_glfw_default OFF)
  endif()
endif()
option(GVD_VIZ_EGL "Headless EGL context (Linux, Mesa llvmpipe in CI)" ${_egl_default})
option(GVD_VIZ_GLFW "GLFW window for live mode (and hidden-window offscreen on Windows)" ${_glfw_default})
option(GVD_VIZ_TESTS "Build the ctest render tests" ON)

include(FetchContent)

# glm and nlohmann_json always come from these pinned, SHA-256-checked
# FetchContent tarballs. System packages are not used, so every builder
# gets the same headers (the PR / spec promise this).
FetchContent_Declare(glm
  URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
  URL_HASH SHA256=9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  SOURCE_SUBDIR gvd_headers_only)  # download only; no add_subdirectory
FetchContent_MakeAvailable(glm)
if(NOT TARGET glm::glm)
  add_library(glm_headers INTERFACE)
  target_include_directories(glm_headers SYSTEM INTERFACE ${glm_SOURCE_DIR})
  add_library(glm::glm ALIAS glm_headers)
endif()

FetchContent_Declare(nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(nlohmann_json)

# GLFW: system package 3.3+ if present, otherwise the pinned 3.4 tarball.
if(GVD_VIZ_GLFW)
  find_package(glfw3 3.3 CONFIG QUIET)
  if(NOT TARGET glfw)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(glfw
      URL https://github.com/glfw/glfw/releases/download/3.4/glfw-3.4.zip
      URL_HASH SHA256=b5ec004b2712fd08e8861dc271428f048775200a2df719ccf575143ba749a3e9
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
    FetchContent_MakeAvailable(glfw)
  endif()
endif()

# Vendored: glad 2 (GL 3.3 core + EGL 1.5, runtime loader) and stb.
add_library(gvd_glad STATIC third_party/glad/src/gl.c)
target_include_directories(gvd_glad PUBLIC third_party/glad/include)
if(GVD_VIZ_EGL)
  target_sources(gvd_glad PRIVATE third_party/glad/src/egl.c)
endif()
if(UNIX)
  target_link_libraries(gvd_glad PUBLIC ${CMAKE_DL_LIBS})
endif()
set_target_properties(gvd_glad PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(gvd_viz_core STATIC
  src/scene.cpp
  src/renderer.cpp
  src/context.cpp
  src/image_io.cpp
  src/udp.cpp)
target_include_directories(gvd_viz_core PUBLIC src third_party/stb)
target_link_libraries(gvd_viz_core PUBLIC gvd_glad glm::glm nlohmann_json::nlohmann_json)
target_compile_definitions(gvd_viz_core PUBLIC GLM_FORCE_RADIANS GLM_ENABLE_EXPERIMENTAL)
if(GVD_VIZ_EGL)
  target_compile_definitions(gvd_viz_core PUBLIC GVD_VIZ_HAVE_EGL=1)
endif()
if(GVD_VIZ_GLFW)
  target_compile_definitions(gvd_viz_core PUBLIC GVD_VIZ_HAVE_GLFW=1)
  target_link_libraries(gvd_viz_core PUBLIC glfw)
endif()
if(WIN32)
  target_link_libraries(gvd_viz_core PUBLIC ws2_32)
  target_compile_definitions(gvd_viz_core PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
endif()
if(MSVC)
  target_compile_options(gvd_viz_core PRIVATE /W4 /utf-8)
else()
  target_compile_options(gvd_viz_core PRIVATE -Wall -Wextra)
endif()

add_executable(gvd_viz src/main.cpp)
target_link_libraries(gvd_viz PRIVATE gvd_viz_core)

if(GVD_VIZ_TESTS)
  enable_testing()
  add_executable(gvd_viz_tests tests/test_render.cpp)
  target_link_libraries(gvd_viz_tests PRIVATE gvd_viz_core)
  add_test(NAME gvd_viz_render COMMAND gvd_viz_tests)
  add_test(NAME gvd_viz_cli_offline
    COMMAND ${CMAKE_COMMAND} -DEXE=$<TARGET_FILE:gvd_viz>
            -DSCENE=${CMAKE_CURRENT_SOURCE_DIR}/tests/scenes/basic.jsonl
            -DOUT=${CMAKE_CURRENT_BINARY_DIR}/cli_offline.png
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cli.cmake)
endif()
