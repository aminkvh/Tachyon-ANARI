cmake_minimum_required(VERSION 3.16)

project(Tachyon VERSION 0.99.6 LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(TACHYON_BUILD_CLI "Build the tachyon command-line renderer" ON)
option(TACHYON_BUILD_ANARI "Build the ANARI device (anari/)" OFF)

find_package(Threads REQUIRED)

# Core ray tracing library (no main()). Source list mirrors build_win/build.sh.
set(TACHYON_CORE_SRCS
  api apigeom box global hash parallel threads camera render trace grid
  bvh intersect sphere plane ring triangle cylinder quadric extvol vol ui
  util vector shade texture coordsys imap light imageio jpeg pngfile ppm
  psd sgirgb tgafile winbmp)

set(TACHYON_CORE_FILES)
foreach(name IN LISTS TACHYON_CORE_SRCS)
  list(APPEND TACHYON_CORE_FILES "src/${name}.c")
endforeach()

add_library(tachyon_core STATIC ${TACHYON_CORE_FILES})
target_include_directories(tachyon_core PUBLIC src)
target_compile_definitions(tachyon_core PUBLIC
  USEPOSIXTHREADS THR USEATOMICBARRIERS _REENTRANT)
set_target_properties(tachyon_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(tachyon_core PUBLIC Threads::Threads)

if(NOT WIN32 AND NOT APPLE)
  target_link_libraries(tachyon_core PUBLIC m)
endif()
if(WIN32)
  target_link_libraries(tachyon_core PUBLIC ws2_32)
endif()

# -O3 is the Release default for GCC/Clang; set it explicitly. No -ffast-math
# or -march=native: output must stay byte-identical across builds.
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(tachyon_core PRIVATE
    $<$<CONFIG:Release>:-O3> -Wall)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT TACHYON_IPO_OK OUTPUT TACHYON_IPO_MSG LANGUAGES C)
if(TACHYON_IPO_OK)
  set_property(TARGET tachyon_core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

if(TACHYON_BUILD_CLI)
  add_executable(tachyon
    demosrc/main.c demosrc/getargs.c demosrc/parse.c demosrc/nffparse.c
    demosrc/ac3dparse.c demosrc/trackball.c
    build_win/stub_spaceball.c)
  target_include_directories(tachyon PRIVATE demosrc)
  target_link_libraries(tachyon PRIVATE tachyon_core)
  if(TACHYON_IPO_OK)
    set_property(TARGET tachyon PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  endif()
endif()

if(TACHYON_BUILD_ANARI)
  add_subdirectory(anari)
endif()
