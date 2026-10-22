// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Thin C++ wrapper around the Tachyon C API. This is the only place that
// includes tachyon.h (its C typedefs such as `vector` and `color` pollute the
// global namespace), so the rest of the device never touches it.

#include <cstddef>
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

 private:
  void destroy();

  struct Style
  {
    std::vector<void *> textures;
    float diffuse{1.f};
  };

  void *m_scene{nullptr};
  std::vector<Style> m_styles;
  void *m_headlight{nullptr};
  bool m_lit{false};
  bool m_transparent{false};
  size_t m_numTriangles{0};
  int m_width{0}, m_height{0};
  float *m_buffer{nullptr};
  float m_lastAspect{-1.f};
  int m_lastProjection{-1};
};

} // namespace tachyon_anari
