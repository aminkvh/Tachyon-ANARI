# Copyright (c) 2026 Tachyon-ANARI contributors
# SPDX-License-Identifier: Apache-2.0
#
# cmake -DINPUT=raytrace.spv -DOUTPUT=raytrace.spv.h -DNAME=kRaytraceSpv -P embed_spirv.cmake
# Writes a shader binary (SPIR-V) or source (Metal, with -DTEXT=ON: line
# endings normalized to LF so the result does not depend on the host) as a C
# array.
file(READ "${INPUT}" hex HEX)
if(TEXT)
  # drop every CR byte
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "\\1;" bytelist "${hex}")
  list(REMOVE_ITEM bytelist "0d")
  string(REPLACE ";" "" hex "${bytelist}")
endif()
string(LENGTH "${hex}" hexlen)
math(EXPR nbytes "${hexlen} / 2")
# 16 bytes (32 hex digits) per line
set(bytes "")
set(pos 0)
while(pos LESS hexlen)
  math(EXPR rest "${hexlen} - ${pos}")
  if(rest GREATER 32)
    set(rest 32)
  endif()
  string(SUBSTRING "${hex}" ${pos} ${rest} chunk)
  string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1," chunk "${chunk}")
  string(APPEND bytes "  ${chunk}\n")
  math(EXPR pos "${pos} + 32")
endwhile()
file(WRITE "${OUTPUT}"
"// Generated from raytrace.slang by slangc (TACHYON_REGENERATE_SHADERS). Do not edit.
#pragma once
alignas(4) static const unsigned char ${NAME}[] = {
${bytes}};
static const unsigned long ${NAME}Size = ${nbytes}ul;
")
