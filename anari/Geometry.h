// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "Sampler.h"
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

// Resolves material parameters that vary per primitive (they come from vertex
// or primitive attributes) to a Tachyon surface style. Geometries call it for
// every primitive with the values of the attributes in 'attrMask'.
struct PrimStyleBinder
{
  virtual ~PrimStyleBinder() = default;
  unsigned attrMask{0}; // bit i: attribute i (4 = color) is needed
  // 'vals[i]': value of attribute i for the primitive (only valid for bits in
  // attrMask), 'prim': index of the primitive. Returns a style id, or < 0 when
  // the primitive is invisible.
  virtual int styleFor(const float4 *vals, size_t prim) = 0;
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
  // 'binder' (may be nullptr): picks the style of each primitive instead of
  // using 'style' for all of them.
  // 'colorSampler' (may be nullptr): maps the values of 'colorAttr' (or, for a
  // primitive sampler, the primitive index; then 'colorAttr' is < 0) to the
  // colors.
  virtual void emit(TachyonScene &,
      int,
      const mat4 &,
      int,
      PrimStyleBinder *,
      const Sampler *) const
  {}
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
      int colorAttr,
      PrimStyleBinder *binder,
      const Sampler *colorSampler) const override;
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

// Common part of the point/segment based geometries: vertex positions plus
// vertex and primitive attributes.
struct AttributedGeometry : public Geometry
{
  AttributedGeometry(TachyonGlobalState *s);

  void commitParameters() override;
  bool isValid() const override { return m_valid; }
  ColorSource attributeSource(int attr) const override;

 protected:
  // Validates the position array (reporting problems for geometry 'what').
  bool validatePositions(const char *what);
  // Validates the attribute arrays for 'numPrims' primitives (dropping the
  // ones that are too small).
  void validateAttributes(size_t numPrims);
  // Component 'c' of (UINT32 or UINT64 based) index element 'i'
  static size_t indexAt(const Array1D &index, size_t i, int comps, int c);

  helium::ChangeObserverPtr<Array1D> m_position;
  std::array<helium::ChangeObserverPtr<Array1D>, kNumAttributes> m_vertexAttr;
  std::array<helium::ChangeObserverPtr<Array1D>, kNumAttributes> m_primAttr;
  bool m_valid{false};
  size_t m_numPrims{0};
};

struct Sphere : public AttributedGeometry
{
  Sphere(TachyonGlobalState *s);

  void commitParameters() override;
  void finalize() override;

  void emit(TachyonScene &scene,
      int style,
      const mat4 &xfm,
      int colorAttr,
      PrimStyleBinder *binder,
      const Sampler *colorSampler) const override;
  bool bounds(const mat4 &xfm, float3 &lo, float3 &hi) const override;

 private:
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_radius; // vertex.radius
  float m_globalRadius{1.f};
};

struct Cylinder : public AttributedGeometry
{
  Cylinder(TachyonGlobalState *s);

  void commitParameters() override;
  void finalize() override;

  void emit(TachyonScene &scene,
      int style,
      const mat4 &xfm,
      int colorAttr,
      PrimStyleBinder *binder,
      const Sampler *colorSampler) const override;
  bool bounds(const mat4 &xfm, float3 &lo, float3 &hi) const override;

 private:
  helium::ChangeObserverPtr<Array1D> m_index;
  helium::ChangeObserverPtr<Array1D> m_radius; // primitive.radius
  helium::ChangeObserverPtr<Array1D> m_vertexCap; // vertex.cap
  float m_globalRadius{1.f};
  int m_globalCaps{0}; // bit 0: first vertex, bit 1: second vertex
};

using UnsupportedGeometry = Unsupported<Geometry>;

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Geometry *, ANARI_GEOMETRY);
}
