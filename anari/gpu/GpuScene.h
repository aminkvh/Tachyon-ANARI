// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// What the GPU backends (Vulkan, Metal) share: the host side scene description
// filled by the add*() calls, its conversion into the buffer layout of
// shaders/raytrace.slang (gather()) and the per frame constants. The backends
// only build acceleration structures, run the shader and read back.

#include "../SceneBackend.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace tachyon_anari {
namespace gpu {

// Matches FrameData in shaders/raytrace.slang
struct FrameData
{
  float posOrtho[4];
  float fwdU0[4];
  float rightDu[4];
  float upV0[4];
  float shade[4];
  float bg[4];
  float unused0[4];
  uint32_t size[4];
  uint32_t counts[4];
  uint32_t flags[4];
  uint32_t styles[4];
};
static_assert(sizeof(FrameData) == 176, "uniform block size");

enum : uint32_t
{
  F_SPHERE_COLORS = 1,
  F_SPHERE_STYLES = 2,
  F_CYL_COLORS = 4,
  F_CYL_STYLES = 8,
  F_SHADOWS = 16,
  F_DEPTH = 32,
  F_ACCUM = 64,
  F_ACCUM_READ = 128,
  F_BG_ALPHA = 256,
  F_TRANSPARENT = 512
};

// Shader bindings (Vulkan descriptor bindings, Metal buffer indices)
enum
{
  BIND_AS = 0,
  BIND_SPHERES = 1,
  BIND_SPHERE_COLORS = 2,
  BIND_SPHERE_STYLES = 3,
  BIND_CYLS = 4,
  BIND_CYL_COLORS = 5,
  BIND_CYL_STYLES = 6,
  BIND_TRI_POS = 7,
  BIND_TRI_SHADE = 8,
  BIND_TRI_AUX = 9,
  BIND_STYLES = 10,
  BIND_SAMPLERS = 11,
  BIND_TEXELS = 12,
  BIND_LIGHTS = 13,
  BIND_OUT = 14,
  BIND_DEPTH = 15,
  BIND_ACCUM = 16,
  BIND_AABBS = 17,
  BIND_INSTANCES = 18,
  BIND_FRAME = 19,
  NUM_BINDINGS = 20
};

inline double nowMs()
{
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch())
      .count();
}

class GpuSceneBase : public SceneBackend
{
 public:
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

  size_t numTriangles() const override { return m_numTriangles; }
  size_t numSpheres() const override { return m_numSpheres; }
  size_t numCylinders() const override { return m_numCylinders; }
  std::vector<std::string> takeWarnings() override;

 protected:
  // 'envPrefix' (e.g. "TACHYON_VULKAN"): the backend's environment variables
  // <prefix>_DEBUG, <prefix>_ACCUM and <prefix>_REFITS
  explicit GpuSceneBase(const char *envPrefix);

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

  // The constants of the next frame (see prepareFrame())
  struct FrameSetup
  {
    FrameData fd;
    std::vector<float> lights; // 8 floats per light
    uint32_t base{0}, count{0}; // first sample index, samples to take
    bool accumulate{false}; // the accumulation buffer is used
    bool depth{false};
    OutputFormat format{OutputFormat::None};
    size_t numPixels{0};
    size_t pixelBytes{0}; // size of the color output (16 when there is none)
  };

  void warnOnce(unsigned id, const char *text);
  // Takes the host description (the prototype arrays are moved out)
  Gathered gather();
  // Camera, lights, flags and progressive refinement state of the frame to
  // render now (call once per rendered frame: it counts identical frames)
  FrameSetup prepareFrame();
  // render() without a scene: background color, infinite depth
  void fillDirectBackground();

  // host side scene description, released by finishBuild()
  std::vector<SurfaceStyle> m_styles;
  std::vector<Proto> m_protos;
  size_t m_cur{0};
  std::vector<Placement> m_placements;
  std::map<const void *, int> m_samplerIds;
  std::vector<SamplerDesc> m_samplers;
  std::vector<Light> m_lights;

  // state of the GPU scene, maintained by the backends
  uint32_t m_sceneFlags{0};
  uint32_t m_styleUniform[2]{0, 0};
  uint64_t m_buildId{0};
  Signature m_sig;
  bool m_haveSig{false};
  bool m_updatable{false}; // the BLASes were built refittable
  unsigned m_refits{0};
  unsigned m_maxRefits{100};

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
  std::string m_logTag; // "[tachyon vulkan]"
  unsigned m_warned{0};
  std::vector<std::string> m_warnings;

 private:
  void ensureColors(Proto &p, bool sphere);
  void ensureStyles(Proto &p, bool sphere);
};

} // namespace gpu
} // namespace tachyon_anari
