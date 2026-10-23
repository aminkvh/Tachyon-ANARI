// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Geometry.h"
#include "Object.h"
#include "Sampler.h"
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
  // Sampler the color is taken from, or nullptr (colorAttribute() is then
  // -1 as well)
  const Sampler *colorSampler() const { return validSampler(m_colorSampler); }

  // Attribute (0-3, 4 = color) a scalar parameter is taken from (its first
  // component), or -1 for the constant value
  int opacityAttribute() const { return m_opacityAttribute; }
  int metallicAttribute() const { return m_metallicAttribute; }
  int roughnessAttribute() const { return m_roughnessAttribute; }
  // Sampler a scalar parameter is taken from (its first component), or
  // nullptr: 0 opacity, 1 metallic, 2 roughness
  const Sampler *scalarSampler(int i) const
  {
    return validSampler(m_scalarSampler[i]);
  }

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
  // 'preview' (the renderer's eyeLight mode) shades the plain color only:
  // opacity, alpha mode and the physically based terms are ignored, as in
  // helide's default renderer.
  bool fillStyle(SurfaceStyle &style,
      float colorAlpha,
      const Scalars *scalars = nullptr,
      bool preview = false) const;

 protected:
  float3 m_color{1.f, 1.f, 1.f};
  float m_colorAlpha{1.f};
  int m_colorAttribute{-1};
  int m_opacityAttribute{-1};
  int m_metallicAttribute{-1};
  int m_roughnessAttribute{-1};
  helium::IntrusivePtr<Sampler> m_colorSampler;
  helium::IntrusivePtr<Sampler> m_scalarSampler[3]; // opacity, metallic, roughness
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
  // A sampler that is unsupported or incomplete is ignored (the constant
  // value is used instead)
  static const Sampler *validSampler(const helium::IntrusivePtr<Sampler> &s)
  {
    return s && s->isValid() ? s.ptr : nullptr;
  }
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
