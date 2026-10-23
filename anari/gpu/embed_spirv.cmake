# Copyright (c) 2026 Tachyon-dev contributors
# SPDX-License-Identifier: Apache-2.0
#
# cmake -DINPUT=raytrace.spv -DOUTPUT=raytrace.spv.h -DNAME=kRaytraceSpv -P embed_spirv.cmake
# Writes a SPIR-V binary as a C array.
file(READ "${INPUT}" hex HEX)
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
