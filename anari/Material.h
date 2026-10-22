// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Geometry.h"
#include "Object.h"
#include "TachyonScene.h"

namespace tachyon_anari {

struct Material : public Object
{
  Material(TachyonGlobalState *s);

  static Material *createInstance(
      std::string_view subtype, TachyonGlobalState *s);

  bool affectsScene() const override { return true; }

  void commitParameters() override;

  // Attribute (0-3, 4 = color) the color is taken from, or -1 for constant
  int colorAttribute() const { return m_colorAttribute; }
  const float3 &color() const { return m_color; }

  // Attribute (0-3, 4 = color) a scalar parameter is taken from (its first
  // component), or -1 for the constant value
  int opacityAttribute() const { return m_opacityAttribute; }
  int metallicAttribute() const { return m_metallicAttribute; }
  int roughnessAttribute() const { return m_roughnessAttribute; }

  // The scalar parameters that can come from attributes
  struct Scalars
  {
    float opacity;
    float metallic;
    float roughness;
  };
  Scalars scalars() const { return {m_opacity, m_metallic, m_roughness}; }

  // Fills the Tachyon surface style (color excluded when it comes from the
  // geometry). 'scalars' (may be nullptr) replaces the constant opacity,
  // metallic and roughness values, e.g. with attribute values. Returns false
  // when the surface is fully transparent.
  bool fillStyle(SurfaceStyle &style,
      float colorAlpha,
      const Scalars *scalars = nullptr) const;

 protected:
  float3 m_color{1.f, 1.f, 1.f};
  float m_colorAlpha{1.f};
  int m_colorAttribute{-1};
  int m_opacityAttribute{-1};
  int m_metallicAttribute{-1};
  int m_roughnessAttribute{-1};
  float m_opacity{1.f};
  float m_alphaCutoff{0.5f};
  enum class AlphaMode
  {
    OPAQUE_,
    BLEND,
    MASK
  } m_alphaMode{AlphaMode::OPAQUE_};
  // physically based only
  bool m_pbr{false};
  float m_metallic{0.f};
  float m_roughness{1.f};

  void readCommon(const char *colorName);
  // Attribute index named by the string parameter 'name' (-1: not a string
  // parameter or no supported attribute)
  int readAttributeName(const char *name);
};

struct Matte : public Material
{
  Matte(TachyonGlobalState *s) : Material(s) {}
  void commitParameters() override;
};

struct PhysicallyBased : public Material
{
  PhysicallyBased(TachyonGlobalState *s) : Material(s) {}
  void commitParameters() override;
};

using UnsupportedMaterial = Unsupported<Material>;

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Material *, ANARI_MATERIAL);
}
