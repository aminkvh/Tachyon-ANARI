// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Metal scene backend (macOS): the Vulkan backend's design on Metal ray
// tracing. Spheres and cylinders are bounding box geometry (boxes computed on
// the CPU), triangles triangle geometry, one primitive acceleration structure
// per primitive type and prototype, one instance acceleration structure with
// an instance per placement, and the compute kernel generated from
// shaders/raytrace.slang (intersection queries). Buffers use shared storage
// (unified memory on Apple silicon). Plain C++ interface: the Objective-C++
// state lives in MetalScene.mm.

#include "../GpuScene.h"

#include <memory>
#include <string>

namespace tachyon_anari {
namespace gpu {

class MetalContext;

class MetalScene : public GpuSceneBase
{
 public:
  // nullptr (with 'reason' set) when there is no Metal device with ray
  // tracing support; throws std::runtime_error on a failure. 'hard': a real
  // failure rather than a missing capability.
  static std::unique_ptr<MetalScene> create(
      std::string &gpuName, std::string &reason, bool &hard);
  // see MetalContext::prefetch()
  static void prefetch();
  static void waitPrefetch();

  ~MetalScene() override;

  void finishBuild() override;
  void render() override;

 private:
  struct Impl;

  explicit MetalScene(std::shared_ptr<MetalContext> ctx);
  void destroyScene();
  void destroyOutputs();
  void uploadSceneData(const Gathered &g);
  void buildSlices(const Gathered &g, bool updatable, bool refitOnly);
  void buildTlas(const Gathered &g);
  void buildFull(Gathered &g, bool updatable);
  void refit(Gathered &g);
  void ensureLightBuffer(size_t lights);

  std::unique_ptr<Impl> m;
};

} // namespace gpu
} // namespace tachyon_anari
