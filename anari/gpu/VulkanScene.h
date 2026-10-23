// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// GPU scene backend (phase 1 spike): spheres in one AABB bottom level
// acceleration structure, one instance in the top level structure, a compute
// shader that traces with ray queries and shades in eyeLight mode.

#include "../SceneBackend.h"
#include "VulkanContext.h"

#include <memory>
#include <string>
#include <vector>

namespace tachyon_anari {
namespace gpu {

class VulkanScene : public SceneBackend
{
 public:
  // nullptr (with 'reason' set) when no capable GPU is available; throws
  // std::runtime_error on a driver failure. 'hard': see VulkanContext.
  static std::unique_ptr<VulkanScene> create(
      std::string &gpuName, std::string &reason, bool &hard);

  ~VulkanScene() override;

  void reset(bool lit, int numThreads) override;
  int addStyle(const SurfaceStyle &style) override;
  void addTriangle(
      int style, const float *p, const float *n, const float *c) override;
  void addSampledTriangle(int style,
      const float *p,
      const float *n,
      const float *vertexAttr,
      AttrSampleFn fn,
      const void *user) override;
  void addSphere(
      int style, const float *center, float radius, const float *rgb) override;
  void addSpheres(int style,
      size_t count,
      const float *center,
      const float *radius,
      float uniformRadius,
      const float *rgb) override;
  void addCylinder(int style,
      const float *p0,
      const float *p1,
      float radius,
      const float *rgb0,
      const float *rgb1,
      int caps) override;
  void addDirectionalLight(const float *direction, const float *rgb) override;
  void addPointLight(const float *position, const float *rgb) override;
  void finishBuild() override;

  void setShading(float ambient, float diffuseScale) override;
  void setBackground(const float *rgb) override;
  void setSamples(int pixelSamples) override;
  void setCamera(const SceneCamera &cam) override;
  void setOutput(int width, int height, float *rgbBuffer) override;
  void setDepthOutput(float *depthBuffer) override;

  void render() override;

  size_t numTriangles() const override { return m_numTriangles; }
  size_t numSpheres() const override { return m_numSpheres; }
  size_t numCylinders() const override { return m_numCylinders; }
  std::vector<std::string> takeWarnings() override;

 private:
  explicit VulkanScene(std::shared_ptr<VulkanContext> ctx);
  void createPipelines();
  void destroyScene();
  void destroyOutputs();
  void warnOnce(int id, const char *text);
  void upload(GpuBuffer &dst, const void *src, size_t bytes);
  void buildAccelerationStructures();
  void ensureColors();
  void updateDescriptors();
  void fillBackground();

  std::shared_ptr<VulkanContext> m_ctx;

  // pipelines (live as long as the scene object)
  VkShaderModule m_module{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_setLayout{VK_NULL_HANDLE};
  VkPipelineLayout m_pipeLayout{VK_NULL_HANDLE};
  VkPipeline m_aabbPipe{VK_NULL_HANDLE};
  VkPipeline m_tracePipe{VK_NULL_HANDLE};
  VkDescriptorPool m_descPool{VK_NULL_HANDLE};
  VkDescriptorSet m_set{VK_NULL_HANDLE};

  // host side scene description, released by finishBuild()
  std::vector<SurfaceStyle> m_styles;
  std::vector<float> m_spheres; // 4 floats per sphere: center, radius
  std::vector<float> m_colors; // 3 floats per sphere, only when needed
  std::vector<std::pair<size_t, int>> m_runs; // (count, style) per addSpheres
  bool m_colorsActive{false};
  float m_baseColor[3]{1.f, 1.f, 1.f};
  size_t m_numSpheresBuilt{0};

  // GPU scene
  GpuBuffer m_sphereBuf, m_colorBuf;
  GpuBuffer m_blasBuf, m_tlasBuf;
  VkAccelerationStructureKHR m_blas{VK_NULL_HANDLE};
  VkAccelerationStructureKHR m_tlas{VK_NULL_HANDLE};
  bool m_descSceneDirty{true};

  // outputs
  GpuBuffer m_outBuf, m_depthBuf, m_readBuf, m_depthReadBuf;
  int m_outW{0}, m_outH{0};
  bool m_outDepth{false};
  bool m_descOutDirty{true};

  // per frame state
  float m_ambient{0.5f}, m_diffuse{0.5f};
  float m_bg[3]{0.f, 0.f, 0.f};
  SceneCamera m_cam;
  int m_width{0}, m_height{0};
  float *m_rgb{nullptr};
  float *m_depth{nullptr};

  size_t m_numTriangles{0}, m_numSpheres{0}, m_numCylinders{0};
  bool m_lit{false};
  bool m_debug{false};
  unsigned m_warned{0};
  std::vector<std::string> m_warnings;
};

} // namespace gpu
} // namespace tachyon_anari
