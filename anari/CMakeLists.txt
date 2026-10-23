# Copyright (c) 2026 Tachyon-ANARI contributors
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

# GPU backends (hardware ray tracing through ray queries): Vulkan on Windows
# and Linux, Metal on macOS. Both run shaders/raytrace.slang, compiled ahead
# of time (the results are committed, so slangc is only needed to change it).
# volk loads Vulkan at run time and Metal is a system framework, so the device
# still works on machines without a capable GPU (it then uses the CPU).
if(APPLE)
  set(_tachyon_vulkan_default OFF)
  set(_tachyon_metal_default ON)
else()
  set(_tachyon_vulkan_default ON)
  set(_tachyon_metal_default OFF)
endif()
option(TACHYON_ANARI_VULKAN "Build the Vulkan GPU backend of the ANARI device"
  ${_tachyon_vulkan_default})
option(TACHYON_ANARI_METAL "Build the Metal GPU backend of the ANARI device (macOS)"
  ${_tachyon_metal_default})
option(TACHYON_REGENERATE_SHADERS
  "Recompile anari/gpu/shaders/*.slang with slangc (the outputs are committed)" OFF)
if(TACHYON_ANARI_METAL AND NOT APPLE)
  message(WARNING "TACHYON_ANARI_METAL needs macOS; building without the Metal backend")
  set(TACHYON_ANARI_METAL OFF CACHE BOOL "" FORCE)
endif()

set(_shader_dir ${CMAKE_CURRENT_SOURCE_DIR}/gpu/shaders)
if(TACHYON_REGENERATE_SHADERS)
  find_program(SLANGC_EXECUTABLE slangc REQUIRED)
endif()

if(TACHYON_ANARI_VULKAN OR TACHYON_ANARI_METAL)
  target_sources(anari_library_tachyon PRIVATE gpu/GpuScene.cpp)
endif()

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
    add_custom_target(tachyon_shaders_spirv DEPENDS ${_shader_dir}/raytrace.spv.h)
    add_dependencies(anari_library_tachyon tachyon_shaders_spirv)
  endif()
endif()

if(TACHYON_ANARI_METAL)
  enable_language(OBJCXX)
  target_sources(anari_library_tachyon PRIVATE
    gpu/metal/MetalContext.mm
    gpu/metal/MetalScene.mm)
  set_source_files_properties(gpu/metal/MetalContext.mm gpu/metal/MetalScene.mm
    PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  target_compile_definitions(anari_library_tachyon PRIVATE TACHYON_ANARI_METAL)
  target_link_libraries(anari_library_tachyon PRIVATE
    "-framework Metal" "-framework Foundation")

  if(TACHYON_REGENERATE_SHADERS)
    # Metal source text: the device compiles it when it starts (no Xcode
    # needed to build or run)
    set(_msl ${CMAKE_CURRENT_BINARY_DIR}/raytrace.metal)
    add_custom_command(
      OUTPUT ${_shader_dir}/raytrace.metal.h
      COMMAND ${SLANGC_EXECUTABLE} ${_shader_dir}/raytrace.slang
        -target metal -DTACHYON_METAL -line-directive-mode none
        -entry main -stage compute
        -o ${_msl}
      COMMAND ${CMAKE_COMMAND} -DINPUT=${_msl}
        -DOUTPUT=${_shader_dir}/raytrace.metal.h -DNAME=kRaytraceMsl -DTEXT=ON
        -P ${CMAKE_CURRENT_SOURCE_DIR}/gpu/embed_spirv.cmake
      DEPENDS ${_shader_dir}/raytrace.slang
        ${CMAKE_CURRENT_SOURCE_DIR}/gpu/embed_spirv.cmake
      COMMENT "Compiling raytrace.slang to Metal source"
      VERBATIM)
    add_custom_target(tachyon_shaders_metal DEPENDS ${_shader_dir}/raytrace.metal.h)
    add_dependencies(anari_library_tachyon tachyon_shaders_metal)
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
