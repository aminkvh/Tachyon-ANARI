// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Renderer-independent scene interface of the ANARI device. Frame and the
// emit() methods of geometries and lights only talk to this; TachyonScene is
// the CPU implementation (Tachyon C core), VulkanScene the GPU one. Nothing in
// here may include tachyon.h or any graphics API header.

#include <cstddef>
#include <cstdint>
#include <string>
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

// What a GPU backend needs to evaluate an ANARI sampler in a shader (the CPU
// backend uses the AttrSampleFn callback instead). Matrices are column major
// (column c is m[4 * c .. 4 * c + 3]); the sampler's result is
//   outTransform * lookup(inTransform * attr + inOffset) + outOffset
// where 'dims' == 0 means there is no lookup (transform sampler).
struct SamplerDesc
{
  int dims{0}; // 0: transform only, 1, 2, 3: image dimensionality
  int size[3]{1, 1, 1};
  int wrap[3]{0, 0, 0}; // 0 clamp to edge, 1 repeat, 2 mirrored repeat
  bool linear{true};
  float inTransform[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  float inOffset[4]{0, 0, 0, 0};
  float outTransform[16]{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  float outOffset[4]{0, 0, 0, 0};
  std::vector<float> texels; // RGBA float per texel, x fastest
};

enum class OutputFormat
{
  None,
  UFixed8, // 4 x uint8, linear
  UFixed8Srgb, // 4 x uint8, sRGB encoded
  Float32 // 4 x float
};

class SceneBackend
{
 public:
  virtual ~SceneBackend() = default;

  // Discard the current scene and start an empty one.
  // 'lit': the application provides lights (otherwise a headlight is used).
  virtual void reset(bool lit, int numThreads) = 0;

  virtual int addStyle(const SurfaceStyle &style) = 0;
  // p: 3 x xyz, n: 3 x xyz or nullptr, c: 3 x rgb or nullptr
  virtual void addTriangle(
      int style, const float *p, const float *n, const float *c) = 0;
  // Triangle whose color is computed for every hit by 'fn(user, attr, rgb)':
  // 'attr' is the 4 component vertex attribute value interpolated at the hit
  // point (the attributes of the 3 vertices are given in 'vertexAttr', 12
  // floats). 'fn' is called from the render threads. The style's color is
  // ignored.
  using AttrSampleFn = void (*)(const void *user, const float *attr, float *rgb);
  virtual void addSampledTriangle(int style,
      const float *p,
      const float *n,
      const float *vertexAttr,
      AttrSampleFn fn,
      const void *user,
      const SamplerDesc *desc) = 0;
  // 'rgb' (3 floats, may be nullptr) overrides the style color.
  virtual void addSphere(
      int style, const float *center, float radius, const float *rgb) = 0;
  // Many spheres at once. 'center': 3 floats per sphere; 'radius': one float
  // per sphere or nullptr for 'uniformRadius'; 'rgb': 3 floats per sphere or
  // nullptr for the style color. The data is copied.
  virtual void addSpheres(int style,
      size_t count,
      const float *center,
      const float *radius,
      float uniformRadius,
      const float *rgb) = 0;
  // Finite cylinder from p0 to p1. 'caps': bit 0 closes the p0 end, bit 1 the
  // p1 end. With two different colors the cylinder is split at its midpoint.
  virtual void addCylinder(int style,
      const float *p0,
      const float *p1,
      float radius,
      const float *rgb0,
      const float *rgb1,
      int caps) = 0;
  // 'direction': the direction the light travels (same as ANARI)
  virtual void addDirectionalLight(const float *direction, const float *rgb) = 0;
  virtual void addPointLight(const float *position, const float *rgb) = 0;
  // Instancing (optional, GPU backends): everything added between
  // beginPrototype() and endPrototype() is one geometry set in object space
  // (the add*() calls then get untransformed data), placed in the scene by
  // addInstance() calls ('xfm': 16 floats, column major 4x4, rotation +
  // translation + uniform scale only). Backends without instancing return
  // false here and the caller bakes the transforms into the geometry.
  virtual bool supportsInstancing() const { return false; }
  virtual int beginPrototype() { return -1; }
  virtual void endPrototype() {}
  virtual void addInstance(int prototype, const float *xfm)
  {
    (void)prototype;
    (void)xfm;
  }
  // Called after the last add*(): builds acceleration structures if needed.
  virtual void finishBuild() = 0;

  // Cheap per-frame updates: none of these rebuild geometry or acceleration
  // structures.
  virtual void setShading(float ambient, float diffuseScale) = 0;
  virtual void setBackground(const float *rgb) = 0;
  virtual void setSamples(int pixelSamples) = 0;
  virtual void setCamera(const SceneCamera &cam) = 0;
  virtual void setOutput(int width, int height, float *rgbBuffer) = 0;
  // Optional per-pixel primary hit distance (same layout as the image, 1e18
  // for misses); nullptr disables it.
  virtual void setDepthOutput(float *depthBuffer) = 0;

  // GPU backends can write the final frame buffers themselves (ANARI row
  // order, bottom row first, in the frame's pixel format). Returns false when
  // unsupported: the caller then uses setOutput()/setDepthOutput() and
  // converts. 'color' (may be nullptr) gets 4 components per pixel, 'depth'
  // (may be nullptr) the primary hit distance (+inf for misses); 'bgAlpha' is
  // the alpha of pixels that show the background.
  virtual bool setDirectOutput(OutputFormat format,
      void *color,
      float *depth,
      float bgAlpha)
  {
    (void)format;
    (void)color;
    (void)depth;
    (void)bgAlpha;
    return false;
  }
  // True when the scene contains something this backend cannot render (the
  // caller may then switch to another backend).
  virtual bool hasUnsupportedContent() const { return false; }

  // Renders into the buffer given to setOutput(), 3 floats per pixel. Rows are
  // stored top row first.
  virtual void render() = 0;

  virtual size_t numTriangles() const = 0;
  virtual size_t numSpheres() const = 0;
  virtual size_t numCylinders() const = 0;

  // Messages about features this backend cannot honour (each reported once
  // per scene); the caller turns them into ANARI warnings.
  virtual std::vector<std::string> takeWarnings() { return {}; }
};

} // namespace tachyon_anari
