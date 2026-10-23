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

// Starts initializing a GPU backend in the background (called when the device
// parameter is committed) so that the first frame only waits for the
// remainder; waitForBackendPrefetch() blocks until that is done (device
// teardown).
void prefetchBackend(const std::string &preference);
void waitForBackendPrefetch();

// 'preference': auto | cpu | vulkan | metal; 'auto' tries Metal, then Vulkan
// (whichever this build has). Never fails: falls back to the CPU backend (with
// a warning when a GPU backend was requested or broke).
BackendChoice createBackend(const std::string &preference);

} // namespace tachyon_anari
