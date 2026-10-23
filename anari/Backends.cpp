// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "Backends.h"

#include "TachyonScene.h"
#ifdef TACHYON_ANARI_VULKAN
#include "gpu/VulkanScene.h"
#endif

#include <exception>

namespace tachyon_anari {

std::unique_ptr<SceneBackend> createCpuBackend()
{
  return std::make_unique<TachyonScene>();
}

void prefetchBackend(const std::string &preference)
{
#ifdef TACHYON_ANARI_VULKAN
  if (preference != "cpu")
    gpu::VulkanContext::prefetch();
#else
  (void)preference;
#endif
}

void waitForBackendPrefetch()
{
#ifdef TACHYON_ANARI_VULKAN
  gpu::VulkanContext::waitPrefetch();
#endif
}

BackendChoice createBackend(const std::string &preference)
{
  BackendChoice c;
  if (preference != "cpu") {
#ifdef TACHYON_ANARI_VULKAN
    std::string gpuName, reason;
    bool hard = false;
    try {
      c.scene = gpu::VulkanScene::create(gpuName, reason, hard);
      if (c.scene) {
        c.gpu = true;
        c.info = "Vulkan backend on " + gpuName;
        return c;
      }
    } catch (const std::exception &e) {
      reason = e.what();
      hard = true;
    }
    if (preference == "vulkan" || hard)
      c.warnings.push_back("Vulkan backend unavailable (" + reason
          + "), using the CPU backend");
    else
      c.info = "CPU backend (no Vulkan ray tracing GPU: " + reason + ")";
#else
    if (preference == "vulkan")
      c.warnings.push_back(
          "this build has no Vulkan backend, using the CPU backend");
#endif
  }
  c.scene = createCpuBackend();
  if (c.info.empty())
    c.info = "CPU backend";
  return c;
}

} // namespace tachyon_anari
