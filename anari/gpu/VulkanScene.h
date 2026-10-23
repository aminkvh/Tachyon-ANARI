// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// GPU scene backend: spheres, cylinders and triangles in bottom level
// acceleration structures (one per primitive type and prototype, AABBs for the
// analytic ones), one top level structure with an instance per placement, and
// a compute shader that traces with ray queries, shades like Tachyon's full
// shader (lights, shadows, Blinn highlights, transparency) and writes the
// frame in its final pixel format.
//
// When consecutive builds have the same topology (the usual MD trajectory
// case: positions change, counts do not) the buffers are updated in place and
// the acceleration structures are refitted instead of rebuilt.

#include "GpuScene.h"
#include "VulkanContext.h"

#include <memory>
#include <string>
#include <vector>

namespace tachyon_anari {
namespace gpu {

class VulkanScene : public GpuSceneBase
{
 public:
  // nullptr (with 'reason' set) when no capable GPU is available; throws
  // std::runtime_error on a driver failure. 'hard': see VulkanContext.
  static std::unique_ptr<VulkanScene> create(
      std::string &gpuName, std::string &reason, bool &hard);

  ~VulkanScene() override;

  void finishBuild() override;
  void render() override;

 private:
  struct Blas
  {
    GpuBuffer buffer;
    VkAccelerationStructureKHR as{VK_NULL_HANDLE};
    VkDeviceAddress address{0};
    VkDeviceSize updateScratch{0};
  };

  explicit VulkanScene(std::shared_ptr<VulkanContext> ctx);
  void createPipelines();
  void destroyScene();
  void destroyOutputs();
  GpuBuffer deviceBuffer(
      const void *data, size_t bytes, VkBufferUsageFlags usage);
  void upload(GpuBuffer &dst, const void *src, size_t bytes);
  // creates the buffer for 'binding' or, when it exists, overwrites it
  void putBuffer(int binding,
      const void *data,
      size_t bytes,
      VkBufferUsageFlags usage);
  void generateAabbs(uint32_t set, uint32_t count, GpuBuffer &aabbs);
  Blas buildBlas(const VkAccelerationStructureGeometryKHR &geom,
      uint32_t prims,
      bool updatable);
  void updateBlas(Blas &blas,
      const VkAccelerationStructureGeometryKHR &geom,
      uint32_t prims);
  void buildTlas(const Gathered &g);
  void destroyBlas(Blas &b);
  void buildFull(Gathered &g, bool updatable);
  void refit(Gathered &g);
  void buildSlices(const Gathered &g, bool updatable, bool refitOnly);
  void uploadSceneData(const Gathered &g);
  void ensureLightBuffer();
  void updateDescriptors();

  std::shared_ptr<VulkanContext> m_ctx;

  VkDescriptorPool m_descPool{VK_NULL_HANDLE};
  VkDescriptorSet m_set{VK_NULL_HANDLE};
  GpuBuffer m_dummy; // bound for unused bindings
  GpuBuffer m_frameUbo; // FrameData, host visible
  GpuBuffer m_lightBuf; // lights, host visible
  size_t m_lightCapacity{0};

  // GPU scene
  GpuBuffer m_buf[20]; // by shader binding (1..18)
  std::vector<Blas> m_blas; // per slice
  GpuBuffer m_tlasBuf;
  VkAccelerationStructureKHR m_tlas{VK_NULL_HANDLE};
  bool m_descDirty{true};

  // outputs
  GpuBuffer m_outBuf, m_depthBuf, m_readBuf, m_depthReadBuf, m_accumBuf;
  int m_outW{0}, m_outH{0};
  size_t m_outBytes{0};
  bool m_outDepth{false};
  OutputFormat m_outFormat{OutputFormat::None};
};

} // namespace gpu
} // namespace tachyon_anari
