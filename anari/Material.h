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

  // Fills the Tachyon surface style (color excluded when it comes from the
  // geometry). Returns false when the surface is fully transparent.
  bool fillStyle(SurfaceStyle &style, float colorAlpha) const;

 protected:
  float3 m_color{1.f, 1.f, 1.f};
  float m_colorAlpha{1.f};
  int m_colorAttribute{-1};
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
