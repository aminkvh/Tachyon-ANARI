// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Thin C++ wrapper around the Tachyon C API. This is the only place that
// includes tachyon.h (its C typedefs such as `vector` and `color` pollute the
// global namespace), so the rest of the device never touches it.

#include "SceneBackend.h"

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace tachyon_anari {

class TachyonScene : public SceneBackend
{
 public:
  TachyonScene();
  ~TachyonScene() override;
  TachyonScene(const TachyonScene &) = delete;
  TachyonScene &operator=(const TachyonScene &) = delete;

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
  // Spheres and cylinders are allocated from the scene's bulk arena (no
  // per-object malloc). Textures are shared between all objects of a style
  // that use the same color, so only the number of distinct colors costs
  // memory.
  void addSphere(
      int style, const float *center, float radius, const float *rgb) override;
  // Many spheres as one Tachyon object with its own compact BVH (about 40
  // bytes per sphere, no per-sphere allocation).
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
  // Adds the camera headlight when the scene is not lit.
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
