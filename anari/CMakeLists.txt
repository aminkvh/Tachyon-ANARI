# Copyright (c) 2026 Tachyon-dev contributors
# SPDX-License-Identifier: Apache-2.0

# ANARI device ("tachyon") that renders triangle scenes with the Tachyon core.

# The top-level project only enables C.
enable_language(CXX)

find_package(anari 0.16 REQUIRED)

add_library(anari_library_tachyon SHARED
  TachyonLibrary.cpp
  TachyonDevice.cpp
  TachyonGlobalState.cpp
  TachyonScene.cpp
  Backends.cpp
  Object.cpp
  Camera.cpp
  Geometry.cpp
  Material.cpp
  Sampler.cpp
  Light.cpp
  World.cpp
  Renderer.cpp
  Frame.cpp
)

# Vulkan GPU backend (hardware ray tracing through ray queries). volk loads
# Vulkan at run time, so the device still works on machines without it. Off on
# macOS by default (no Vulkan ray tracing there; a Metal backend will follow).
if(APPLE)
  set(_tachyon_vulkan_default OFF)
else()
  set(_tachyon_vulkan_default ON)
endif()
option(TACHYON_ANARI_VULKAN "Build the Vulkan GPU backend of the ANARI device"
  ${_tachyon_vulkan_default})
option(TACHYON_REGENERATE_SHADERS
  "Recompile anari/gpu/shaders/*.slang with slangc (the SPIR-V is committed)" OFF)

if(TACHYON_ANARI_VULKAN)
  include(FetchContent)
  # Pinned: Vulkan-Headers v1.4.363 and the volk release made for it
  # (vulkan-sdk-1.4.363.0). No Vulkan SDK is needed to build.
  FetchContent_Declare(vulkan_headers
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG v1.4.363
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL)
  FetchContent_Declare(volk
    GIT_REPOSITORY https://github.com/zeux/volk.git
    GIT_TAG vulkan-sdk-1.4.363.0
    GIT_SHALLOW TRUE
    EXCLUDE_FROM_ALL)
  FetchContent_MakeAvailable(vulkan_headers volk)
  set_target_properties(volk PROPERTIES POSITION_INDEPENDENT_CODE ON)

  target_sources(anari_library_tachyon PRIVATE
    gpu/VulkanContext.cpp
    gpu/VulkanScene.cpp)
  target_compile_definitions(anari_library_tachyon PRIVATE TACHYON_ANARI_VULKAN)
  target_link_libraries(anari_library_tachyon PRIVATE volk Vulkan::Headers)

  if(TACHYON_REGENERATE_SHADERS)
    find_program(SLANGC_EXECUTABLE slangc REQUIRED)
    set(_shader_dir ${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders)
    set(_spv ${CMAKE_CURRENT_BINARY_DIR}/raytrace.spv)
    add_custom_command(
      OUTPUT ${_shader_dir}/raytrace.spv.h
      COMMAND ${SLANGC_EXECUTABLE} ${_shader_dir}/raytrace.slang
        -target spirv -profile spirv_1_4 -capability spvRayQueryKHR
        -entry genAabbs -stage compute -entry main -stage compute
        -o ${_spv}
      COMMAND ${CMAKE_COMMAND} -DINPUT=${_spv}
        -DOUTPUT=${_shader_dir}/raytrace.spv.h -DNAME=kRaytraceSpv
        -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu/embed_spirv.cmake
      DEPENDS ${_shader_dir}/raytrace.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/embed_spirv.cmake
      COMMENT "Compiling raytrace.slang to SPIR-V"
      VERBATIM)
    add_custom_target(tachyon_shaders DEPENDS ${_shader_dir}/raytrace.spv.h)
    add_dependencies(anari_library_tachyon tachyon_shaders)
  endif()
endif()

# Generated ANARI object/parameter/extension query tables.
anari_generate_queries(
  DEVICE_TARGET anari_library_tachyon
  CPP_NAMESPACE tachyon_anari
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/TachyonDefinitions.json
)

include(GenerateExportHeader)
generate_export_header(anari_library_tachyon EXPORT_MACRO_NAME "TACHYON_ANARI_EXPORT")

target_include_directories(anari_library_tachyon PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(anari_library_tachyon
  PUBLIC anari::helium
  PRIVATE tachyon_core)

set_target_properties(anari_library_tachyon PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(anari_library_tachyon PRIVATE -Wall)
endif()
if(WIN32)
  target_compile_definitions(anari_library_tachyon PRIVATE _USE_MATH_DEFINES)
endif()

include(GNUInstallDirs)
install(TARGETS anari_library_tachyon
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

# The ANARI loader opens anari_library_<name>.dll on Windows; MinGW would
# otherwise add a "lib" prefix.
if(WIN32)
  set_target_properties(anari_library_tachyon PROPERTIES PREFIX "")
endif()
