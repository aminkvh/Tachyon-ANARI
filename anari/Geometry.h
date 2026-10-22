// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "TachyonScene.h"

namespace tachyon_anari {

// Where the colors for a material with color="color" come from.
enum class ColorSource
{
  NONE,
  VERTEX,
  PRIMITIVE,
  UNIFORM
};

struct Geometry : public Object
{
  Geometry(TachyonGlobalState *s);

  static Geometry *createInstance(
      std::string_view subtype, TachyonGlobalState *s);

  bool affectsScene() const override { return true; }

  void commitParameters() override;

  // Which color source is available (used by materials with color="color")
  virtual ColorSource colorSource() const { return ColorSource::NONE; }
  const float4 &uniformColor() const { return m_uniformColor; }

  // Feeds the geometry, transformed by 'xfm', to the Tachyon scene.
  // 'useColors': emit per-vertex/-primitive colors (see colorSource()).
  virtual void emit(TachyonScene &, int, const mat4 &, bool) const {}
  virtual bool bounds(const mat4 &, float3 &, float3 &) const { return false; }

 protected:
  bool m_hasUniformColor{false};
  float4 m_uniformColor{1.f, 1.f, 1.f, 1.f};
};

struct Triangle : public Geometry
{
  Triangle(TachyonGlobalState *s);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override { return m_valid; }

  ColorSource colorSource() const override;
  void emit(TachyonScene &scene,
      int style,
      const mat4 &xfm,
      bool useColors) const override;
  bool bounds(const mat4 &xfm, float3 &lo, float3 &hi) const override;

 private:
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_position;
  helium::ChangeObserverPtr<Array1D> m_normal;
  helium::ChangeObserverPtr<Array1D> m_vertexColor;
  helium::ChangeObserverPtr<Array1D> m_primitiveColor;
  bool m_valid{false};
  size_t m_numTriangles{0};
};

using UnsupportedGeometry = Unsupported<Geometry>;

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Geometry *, ANARI_GEOMETRY);
}
