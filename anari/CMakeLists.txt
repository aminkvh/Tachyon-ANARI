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
  Object.cpp
  Camera.cpp
  Geometry.cpp
  Material.cpp
  Light.cpp
  World.cpp
  Renderer.cpp
  Frame.cpp
)

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
