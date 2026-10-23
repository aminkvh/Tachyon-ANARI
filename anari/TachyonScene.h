// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Thin C++ wrapper around the Tachyon C API. This is the only place that
// includes tachyon.h (its C typedefs such as `vector` and `color` pollute the
// global namespace), so the rest of the device never touches it.

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tachyon_anari {

struct SurfaceStyle
{
  float color[3]{1.f, 1.f, 1.f};
  float opacity{1.f};
  float diffuse{1.f}; // scales the frame-wide diffuse weight
  float phong{0.f}; // highlight strength (0: none)
  float phongExp{10.f};
  bool metal{false};
};

struct SceneCamera
{
  bool orthographic{false};
  float position[3]{0.f, 0.f, 0.f};
  float direction[3]{0.f, 0.f, 1.f};
  float up[3]{0.f, 1.f, 0.f};
  float fovy{1.0471976f}; // radians (perspective)
  float height{1.f}; // world units (orthographic)
  float aspect{1.f}; // image plane width / height
  float region[4]{0.f, 0.f, 1.f, 1.f}; // imageRegion x0,y0,x1,y1 (y up)
};

class TachyonScene
{
 public:
  TachyonScene();
  ~TachyonScene();
  TachyonScene(const TachyonScene &) = delete;
  TachyonScene &operator=(const TachyonScene &) = delete;

  // Discard the current Tachyon scene and start an empty one.
  // 'lit': the application provides lights (otherwise a headlight is used).
  void reset(bool lit, int numThreads);

  int addStyle(const SurfaceStyle &style);
  // p: 3 x xyz, n: 3 x xyz or nullptr, c: 3 x rgb or nullptr
  void addTriangle(int style, const float *p, const float *n, const float *c);
  // Triangle whose color is computed for every hit by 'fn(user, attr, rgb)':
  // 'attr' is the 4 component vertex attribute value interpolated at the hit
  // point (the attributes of the 3 vertices are given in 'vertexAttr', 12
  // floats). 'fn' is called from the render threads. The style's color is
  // ignored.
  using AttrSampleFn = void (*)(const void *user, const float *attr, float *rgb);
  void addSampledTriangle(int style,
      const float *p,
      const float *n,
      const float *vertexAttr,
      AttrSampleFn fn,
      const void *user);
  // Spheres and cylinders are allocated from the scene's bulk arena (no
  // per-object malloc). 'rgb' (3 floats, may be nullptr) overrides the style
  // color; textures are shared between all objects of a style that use the
  // same color, so only the number of distinct colors costs memory.
  void addSphere(int style, const float *center, float radius, const float *rgb);
  // Many spheres as one Tachyon object with its own compact BVH (about 40
  // bytes per sphere, no per-sphere allocation). 'center': 3 floats per
  // sphere; 'radius': one float per sphere or nullptr for 'uniformRadius';
  // 'rgb': 3 floats per sphere or nullptr for the style color. The data is
  // copied.
  void addSpheres(int style,
      size_t count,
      const float *center,
      const float *radius,
      float uniformRadius,
      const float *rgb);
  // Finite cylinder from p0 to p1. 'caps': bit 0 closes the p0 end, bit 1 the
  // p1 end. With two different colors the cylinder is split at its midpoint
  // (Tachyon has no color gradients), one half per color.
  void addCylinder(int style,
      const float *p0,
      const float *p1,
      float radius,
      const float *rgb0,
      const float *rgb1,
      int caps);
  // 'direction': the direction the light travels (same as ANARI)
  void addDirectionalLight(const float *direction, const float *rgb);
  void addPointLight(const float *position, const float *rgb);
  // Adds the camera headlight when the scene is not lit.
  void finishBuild();

  // Cheap per-frame updates: none of these rebuild geometry or acceleration
  // structures.
  void setShading(float ambient, float diffuseScale);
  void setBackground(const float *rgb);
  void setSamples(int pixelSamples);
  void setCamera(const SceneCamera &cam);
  void setOutput(int width, int height, float *rgbBuffer);
  // Optional per-pixel primary hit distance (same layout as the image, 1e18
  // for misses); nullptr disables it.
  void setDepthOutput(float *depthBuffer);

  // Renders into the buffer given to setOutput(), 3 floats per pixel. Rows are
  // stored top row first.
  void render();

  size_t numTriangles() const { return m_numTriangles; }
  size_t numSpheres() const { return m_numSpheres; }
  size_t numCylinders() const { return m_numCylinders; }

 private:
  void destroy();

  struct ColorKey
  {
    uint32_t bits[3];
    bool operator==(const ColorKey &o) const
    {
      return bits[0] == o.bits[0] && bits[1] == o.bits[1] && bits[2] == o.bits[2];
    }
  };
  struct ColorKeyHash
  {
    size_t operator()(const ColorKey &k) const
    {
      uint64_t h = k.bits[0] * 0x9E3779B97F4A7C15ull;
      h = (h ^ (h >> 29)) + k.bits[1] * 0xBF58476D1CE4E5B9ull;
      h = (h ^ (h >> 29)) + k.bits[2] * 0x94D049BB133111EBull;
      return size_t(h ^ (h >> 32));
    }
  };
  struct Style
  {
    std::vector<void *> textures; // [0]: base; the rest need shading updates
    float diffuse{1.f};
    std::unordered_map<ColorKey, void *, ColorKeyHash> colorTextures;
    ColorKey lastKey{{0, 0, 0}};
    void *lastTex{nullptr}; // one-entry cache: runs of equal colors are common
  };

  void *colorTexture(Style &st, const float *rgb);

  void *m_scene{nullptr};
  std::vector<Style> m_styles;
  void *m_headlight{nullptr};
  bool m_lit{false};
  bool m_transparent{false};
  size_t m_numTriangles{0};
  size_t m_numSpheres{0};
  size_t m_numCylinders{0};
  int m_width{0}, m_height{0};
  float *m_buffer{nullptr};
  float m_lastAspect{-1.f};
  int m_lastProjection{-1};
};

} // namespace tachyon_anari
