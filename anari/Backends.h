// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "SceneBackend.h"

#include <memory>
#include <string>
#include <vector>

namespace tachyon_anari {

struct BackendChoice
{
  std::unique_ptr<SceneBackend> scene;
  bool gpu{false};
  std::string info; // e.g. "Vulkan backend on <GPU name>"
  std::vector<std::string> warnings;
};

std::unique_ptr<SceneBackend> createCpuBackend();

// 'preference': auto | cpu | vulkan. Never fails: falls back to the CPU
// backend (with a warning when a GPU backend was requested or broke).
BackendChoice createBackend(const std::string &preference);

} // namespace tachyon_anari
