// Copyright (c) 2026 Tachyon-dev contributors
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
  bool supportsInstancing() const override { return true; }
  int beginPrototype() override;
  void endPrototype() override;
  void addInstance(int prototype, const float *xfm) override;
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
    VkDeviceSize updateScratch{0};
  };

  struct Run
  {
    size_t count;
    int style;
  };

  // Everything added between beginPrototype() and endPrototype() (prototype 0
  // is the scene itself)
  struct Proto
  {
    std::vector<float> spheres; // 4 floats per sphere: center, radius
    std::vector<float> sphereColors; // 3 per sphere, only when needed
    std::vector<uint32_t> sphereStyles; // only when styles differ
    std::vector<Run> sphereRuns;
    bool sphereColorsActive{false}, sphereStylesActive{false};
    int sphereStyle0{0};

    std::vector<float> cyls; // 8 floats per cylinder: p0, r, p1, caps (bits)
    std::vector<float> cylColors; // 6 per cylinder (color at p0, at p1)
    std::vector<uint32_t> cylStyles;
    std::vector<Run> cylRuns;
    bool cylColorsActive{false}, cylStylesActive{false};
    int cylStyle0{0};

    std::vector<float> triPos; // 9 per triangle
    std::vector<float> triShade; // 12 per triangle
    std::vector<float> triAux; // vertex colors (9) / attributes (12)
  };

  struct Placement
  {
    int proto;
    float m[12]; // object to world, rows of a 3x4 matrix
  };

  struct Light
  {
    float v[3]; // direction to the light | position
    int type; // 0 directional, 1 point
    float color[3];
  };

  // A run of primitives of one type belonging to one prototype: one BLAS
  struct Slice
  {
    int proto;
    int type; // 0 spheres, 1 cylinders, 2 triangles
    uint32_t base;
    uint32_t count;
  };
  struct InstanceEntry
  {
    int slice;
    float m[12];
    uint32_t flags; // 1 identity, 2 negative determinant
  };

  // What decides whether a new build can refit the existing structures
  struct Signature
  {
    std::vector<uint32_t> layout;
    std::vector<float> styles;
    uint32_t flags{0};
    size_t aux{0}, texels{0}, samplers{0};
    bool operator==(const Signature &o) const
    {
      return layout == o.layout && styles == o.styles && flags == o.flags
          && aux == o.aux && texels == o.texels && samplers == o.samplers;
    }
  };

  // The scene in its final GPU layout
  struct Gathered
  {
    std::vector<float> spheres, sphereColors, cyls, cylColors;
    std::vector<uint32_t> sphereStyles, cylStyles;
    std::vector<float> triPos, triShade, triAux;
    std::vector<float> styles, samplerTable, texels, instTable;
    std::vector<Slice> slices;
    std::vector<InstanceEntry> instances;
    uint32_t flags{0};
    uint32_t styleUniform[2]{0, 0};
    uint32_t nSph{0}, nCyl{0}, nTri{0};
    Signature sig;
  };

  explicit VulkanScene(std::shared_ptr<VulkanContext> ctx);
  void createPipelines();
  void destroyScene();
  void destroyOutputs();
  void warnOnce(unsigned id, const char *text);
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
  void ensureColors(Proto &p, bool sphere);
  void ensureStyles(Proto &p, bool sphere);
  Gathered gather();
  void buildFull(Gathered &g, bool updatable);
  void refit(Gathered &g);
  void buildSlices(const Gathered &g, bool updatable, bool refitOnly);
  void uploadSceneData(const Gathered &g);
  void ensureLightBuffer();
  void updateDescriptors();
  void fillDirectBackground();

  std::shared_ptr<VulkanContext> m_ctx;

  VkDescriptorPool m_descPool{VK_NULL_HANDLE};
  VkDescriptorSet m_set{VK_NULL_HANDLE};
  GpuBuffer m_dummy; // bound for unused bindings
  GpuBuffer m_frameUbo; // FrameData, host visible
  GpuBuffer m_lightBuf; // lights, host visible
  size_t m_lightCapacity{0};

  // host side scene description, released by finishBuild()
  std::vector<SurfaceStyle> m_styles;
  std::vector<Proto> m_protos;
  size_t m_cur{0};
  std::vector<Placement> m_placements;
  std::map<const void *, int> m_samplerIds;
  std::vector<SamplerDesc> m_samplers;
  std::vector<Light> m_lights;

  // GPU scene
  GpuBuffer m_buf[20]; // by shader binding (1..18)
  std::vector<Blas> m_blas; // per slice
  GpuBuffer m_tlasBuf;
  VkAccelerationStructureKHR m_tlas{VK_NULL_HANDLE};
  uint32_t m_sceneFlags{0};
  uint32_t m_styleUniform[2]{0, 0};
  uint64_t m_buildId{0};
  bool m_descDirty{true};
  Signature m_sig;
  bool m_haveSig{false};
  bool m_updatable{false}; // the BLASes were built with ALLOW_UPDATE
  unsigned m_refits{0};
  unsigned m_maxRefits{100};

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
