// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "helium/BaseGlobalDeviceState.h"
#include "helium/helium_math.h"
// std
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>

namespace tachyon_anari {

using namespace anari::math;
using mat4 = anari::math::float4x4;

struct TachyonGlobalState : public helium::BaseGlobalDeviceState
{
  // Bumped whenever an object that influences the Tachyon scene (geometry,
  // material, surface, group, instance, world, light) is finalized. Frames
  // rebuild the Tachyon scene only when this is newer than their last build.
  std::atomic<helium::TimeStamp> sceneStamp{1};

  int numThreads{-1}; // <= 0: one per CPU

  // "backend" device parameter: auto | cpu | vulkan
  std::string backend{"auto"};

  anari::Device anariDevice{nullptr};

  TachyonGlobalState(ANARIDevice d);
};

} // namespace tachyon_anari
