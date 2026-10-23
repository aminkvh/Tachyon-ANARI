// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// GPU scene backend: spheres, cylinders and triangles in bottom level
// acceleration structures (one per primitive type, AABBs for the analytic
// ones), one top level structure, and a compute shader that traces with ray
// queries, shades like Tachyon's full shader (lights, shadows, Blinn
// highlights, transparency) and writes the frame in its final pixel format.

#include "../SceneBackend.h"
#include "VulkanContext.h"

#include <map>
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
      const void *user,
      const SamplerDesc *desc) override;
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
  bool setDirectOutput(OutputFormat format,
      void *color,
      float *depth,
      float bgAlpha) override;
  bool hasUnsupportedContent() const override { return m_unsupported; }

  void render() override;

  size_t numTriangles() const override { return m_numTriangles; }
  size_t numSpheres() const override { return m_numSpheres; }
  size_t numCylinders() const override { return m_numCylinders; }
  std::vector<std::string> takeWarnings() override;

 private:
  struct Blas
  {
    GpuBuffer buffer;
    VkAccelerationStructureKHR as{VK_NULL_HANDLE};
    VkDeviceAddress address{0};
  };

  explicit VulkanScene(std::shared_ptr<VulkanContext> ctx);
  void createPipelines();
  void destroyScene();
  void destroyOutputs();
  void warnOnce(unsigned id, const char *text);
  GpuBuffer deviceBuffer(
      const void *data, size_t bytes, VkBufferUsageFlags usage);
  void upload(GpuBuffer &dst, const void *src, size_t bytes);
  void generateAabbs(uint32_t set, uint32_t count, GpuBuffer &aabbs);
  Blas buildBlas(const VkAccelerationStructureGeometryKHR &geom, uint32_t prims);
  void buildTlas(const Blas *blas, const uint32_t *types, int count);
  void destroyBlas(Blas &b);
  void ensureSphereColors();
  void ensureSphereStyles();
  void ensureCylColors();
  void ensureCylStyles();
  void updateDescriptors();
  void fillDirectBackground();

  std::shared_ptr<VulkanContext> m_ctx;

  // pipelines (live as long as the scene object)
  VkShaderModule m_module{VK_NULL_HANDLE};
  VkDescriptorSetLayout m_setLayout{VK_NULL_HANDLE};
  VkPipelineLayout m_pipeLayout{VK_NULL_HANDLE};
  VkPipeline m_aabbPipe{VK_NULL_HANDLE};
  VkPipeline m_tracePipe{VK_NULL_HANDLE};
  VkDescriptorPool m_descPool{VK_NULL_HANDLE};
  VkDescriptorSet m_set{VK_NULL_HANDLE};
  GpuBuffer m_dummy; // bound for unused bindings
  GpuBuffer m_frameUbo; // FrameData, host visible
  GpuBuffer m_lightBuf; // lights, host visible
  size_t m_lightCapacity{0};

  // host side scene description, released by finishBuild()
  struct Run
  {
    size_t count;
    int style;
  };
  std::vector<SurfaceStyle> m_styles;

  std::vector<float> m_spheres; // 4 floats per sphere: center, radius
  std::vector<float> m_sphereColors; // 3 per sphere, only when needed
  std::vector<uint32_t> m_sphereStyles; // only when styles differ
  std::vector<Run> m_sphereRuns;
  bool m_sphereColorsActive{false}, m_sphereStylesActive{false};
  int m_sphereStyle0{0};

  std::vector<float> m_cyls; // 8 floats per cylinder: p0, r, p1, caps (bits)
  std::vector<float> m_cylColors; // 6 per cylinder (color at p0, at p1)
  std::vector<uint32_t> m_cylStyles;
  std::vector<Run> m_cylRuns;
  bool m_cylColorsActive{false}, m_cylStylesActive{false};
  int m_cylStyle0{0};

  std::vector<float> m_triPos; // 9 per triangle
  std::vector<float> m_triShade; // 12 per triangle
  std::vector<float> m_triAux; // vertex colors (9) / attributes (12)
  std::map<const void *, int> m_samplerIds;
  std::vector<SamplerDesc> m_samplers;

  struct Light
  {
    float v[3]; // direction to the light | position
    int type; // 0 directional, 1 point
    float color[3];
  };
  std::vector<Light> m_lights;

  // GPU scene
  GpuBuffer m_buf[19]; // by shader binding (1..17)
  Blas m_blasSphere, m_blasCyl, m_blasTri;
  GpuBuffer m_tlasBuf;
  VkAccelerationStructureKHR m_tlas{VK_NULL_HANDLE};
  uint32_t m_counts[3]{0, 0, 0}; // spheres, cylinders, triangles built
  uint32_t m_sceneFlags{0};
  uint32_t m_styleUniform[2]{0, 0};
  uint64_t m_buildId{0};
  bool m_descDirty{true};

  // outputs
  GpuBuffer m_outBuf, m_depthBuf, m_readBuf, m_depthReadBuf, m_accumBuf;
  int m_outW{0}, m_outH{0};
  size_t m_outBytes{0};
  bool m_outDepth{false};
  OutputFormat m_outFormat{OutputFormat::None};

  // per frame state
  float m_ambient{0.5f}, m_diffuse{0.5f};
  float m_bg[4]{0.f, 0.f, 0.f, 1.f};
  SceneCamera m_cam;
  int m_width{0}, m_height{0};
  int m_samples{1};
  OutputFormat m_format{OutputFormat::None};
  void *m_colorDst{nullptr};
  float *m_depthDst{nullptr};
  bool m_accumEnabled{true};
  uint64_t m_lastFrameHash{0};
  uint32_t m_sameFrames{0};

  size_t m_numTriangles{0}, m_numSpheres{0}, m_numCylinders{0};
  bool m_lit{false};
  bool m_transparent{false}; // a style has opacity < 1 (the CPU full shader)
  bool m_unsupported{false};
  bool m_debug{false};
  unsigned m_warned{0};
  std::vector<std::string> m_warnings;
};

} // namespace gpu
} // namespace tachyon_anari
