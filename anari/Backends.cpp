// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "Backends.h"

#include "TachyonScene.h"
#ifdef TACHYON_ANARI_VULKAN
#include "gpu/VulkanScene.h"
#endif
#ifdef TACHYON_ANARI_METAL
#include "gpu/metal/MetalScene.h"
#endif

#include <exception>

namespace tachyon_anari {

std::unique_ptr<SceneBackend> createCpuBackend()
{
  return std::make_unique<TachyonScene>();
}

void prefetchBackend(const std::string &preference)
{
#ifdef TACHYON_ANARI_METAL
  if (preference == "auto" || preference == "metal")
    gpu::MetalScene::prefetch();
#endif
#ifdef TACHYON_ANARI_VULKAN
  if (preference == "auto" || preference == "vulkan")
    gpu::VulkanContext::prefetch();
#endif
  (void)preference;
}

void waitForBackendPrefetch()
{
#ifdef TACHYON_ANARI_METAL
  gpu::MetalScene::waitPrefetch();
#endif
#ifdef TACHYON_ANARI_VULKAN
  gpu::VulkanContext::waitPrefetch();
#endif
}

namespace {

// Tries one GPU backend. 'requested': the application asked for exactly this
// one (a failure is then a warning, not an INFO note).
template <class Scene>
bool tryGpu(BackendChoice &c,
    const char *api,
    bool requested,
    std::string &note)
{
  std::string gpuName, reason;
  bool hard = false;
  try {
    c.scene = Scene::create(gpuName, reason, hard);
    if (c.scene) {
      c.gpu = true;
      c.info = std::string(api) + " backend on " + gpuName;
      return true;
    }
  } catch (const std::exception &e) {
    reason = e.what();
    hard = true;
  }
  if (requested || hard)
    c.warnings.push_back(std::string(api) + " backend unavailable (" + reason
        + "), using the CPU backend");
  else
    note = std::string("no ") + api + " ray tracing GPU: " + reason;
  return false;
}

} // namespace

BackendChoice createBackend(const std::string &preference)
{
  BackendChoice c;
  std::string note;
  const bool any = preference == "auto";
  (void)any;
#ifdef TACHYON_ANARI_METAL
  if ((any || preference == "metal")
      && tryGpu<gpu::MetalScene>(c, "Metal", preference == "metal", note))
    return c;
#else
  if (preference == "metal")
    c.warnings.push_back(
        "this build has no Metal backend, using the CPU backend");
#endif
#ifdef TACHYON_ANARI_VULKAN
  if ((any || preference == "vulkan")
      && tryGpu<gpu::VulkanScene>(c, "Vulkan", preference == "vulkan", note))
    return c;
#else
  if (preference == "vulkan")
    c.warnings.push_back(
        "this build has no Vulkan backend, using the CPU backend");
#endif
  c.scene = createCpuBackend();
  c.gpu = false;
  c.info = note.empty() ? "CPU backend" : "CPU backend (" + note + ")";
  return c;
}

} // namespace tachyon_anari
