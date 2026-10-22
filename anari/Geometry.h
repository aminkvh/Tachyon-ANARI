// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "TachyonScene.h"
// std
#include <array>
#include <optional>

namespace tachyon_anari {

constexpr int kNumAttributes = 5; // attribute0-3 and color

// Where the values of an attribute come from.
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

  // Which source provides attribute 'attr' (0-3, 4 = color), following the
  // lookup order vertex > primitive > uniform
  virtual ColorSource attributeSource(int attr) const
  {
    return m_uniform[attr] ? ColorSource::UNIFORM : ColorSource::NONE;
  }
  const std::optional<float4> &uniformAttribute(int attr) const
  {
    return m_uniform[attr];
  }

  // Feeds the geometry, transformed by 'xfm', to the Tachyon scene.
  // 'colorAttr' >= 0: emit per-vertex/-primitive values of that attribute as
  // colors (see attributeSource()); < 0: no per-primitive colors.
  virtual void emit(TachyonScene &, int, const mat4 &, int) const {}
  virtual bool bounds(const mat4 &, float3 &, float3 &) const { return false; }

 protected:
  std::array<std::optional<float4>, kNumAttributes> m_uniform;
};

struct Triangle : public Geometry
{
  Triangle(TachyonGlobalState *s);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override { return m_valid; }

  ColorSource attributeSource(int attr) const override;
  void emit(TachyonScene &scene,
      int style,
      const mat4 &xfm,
      int colorAttr) const override;
  bool bounds(const mat4 &xfm, float3 &lo, float3 &hi) const override;

 private:
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_position;
  helium::ChangeObserverPtr<Array1D> m_normal;
  std::array<helium::ChangeObserverPtr<Array1D>, kNumAttributes> m_vertexAttr;
  std::array<helium::ChangeObserverPtr<Array1D>, kNumAttributes> m_primAttr;
  bool m_valid{false};
  size_t m_numTriangles{0};
};

using UnsupportedGeometry = Unsupported<Geometry>;

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Geometry *, ANARI_GEOMETRY);
}
