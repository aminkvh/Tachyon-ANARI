// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "Material.h"
// std
#include <algorithm>
#include <cmath>

namespace tachyon_anari {

Material::Material(TachyonGlobalState *s) : Object(ANARI_MATERIAL, s) {}

Material *Material::createInstance(
    std::string_view subtype, TachyonGlobalState *s)
{
  if (subtype == "matte")
    return new Matte(s);
  if (subtype == "physicallyBased")
    return new PhysicallyBased(s);
  return nullptr; // caller creates an UnsupportedMaterial
}

void Material::commitParameters()
{
  const std::string mode = getParamString("alphaMode", "opaque");
  m_alphaMode = mode == "blend"
      ? AlphaMode::BLEND
      : (mode == "mask" ? AlphaMode::MASK : AlphaMode::OPAQUE_);
  m_alphaCutoff = getParam<float>("alphaCutoff", 0.5f);
  m_opacity = getParam<float>("opacity", 1.f);
}

void Material::readCommon(const char *colorName)
{
  m_color = float3(1.f);
  m_colorAlpha = 1.f;
  float4 c4;
  float3 c3;
  if (getParam(colorName, ANARI_FLOAT32_VEC4, &c4)) {
    m_color = float3(c4.x, c4.y, c4.z);
    m_colorAlpha = c4.w;
  } else if (getParam(colorName, ANARI_FLOAT32_VEC3, &c3)) {
    m_color = c3;
  }

  const std::string attr = getParamString(colorName, "none");
  m_colorFromGeometry = attr == "color";
  if (attr != "none" && attr != "color" && !attr.empty()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "color attribute '%s' not supported, using constant color",
        attr.c_str());
  }
  if (getParamObject<Object>(colorName)) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "samplers are not supported, using constant color");
  }
}

bool Material::fillStyle(SurfaceStyle &s, float colorAlpha) const
{
  if (!m_colorFromGeometry) {
    s.color[0] = m_color.x;
    s.color[1] = m_color.y;
    s.color[2] = m_color.z;
  }

  float alpha = m_opacity * colorAlpha;
  switch (m_alphaMode) {
  case AlphaMode::OPAQUE_:
    alpha = 1.f;
    break;
  case AlphaMode::MASK:
    alpha = alpha >= m_alphaCutoff ? 1.f : 0.f;
    break;
  case AlphaMode::BLEND:
    break;
  }
  s.opacity = std::clamp(alpha, 0.f, 1.f);

  if (m_pbr) {
    // Rough approximation: metals lose diffuse reflection and gain a tinted
    // highlight; rough surfaces get a wide, weak highlight.
    const float m = std::clamp(m_metallic, 0.f, 1.f);
    const float r = std::clamp(m_roughness, 0.f, 1.f);
    s.diffuse = 1.f - 0.7f * m;
    s.phong = (0.2f + 0.7f * m) * (1.f - 0.7f * r);
    s.phongExp = std::clamp(2.f / (r * r + 0.002f), 4.f, 400.f);
    s.metal = m > 0.5f;
  }
  return s.opacity > 0.f;
}

void Matte::commitParameters()
{
  Material::commitParameters();
  m_pbr = false;
  readCommon("color");
}

void PhysicallyBased::commitParameters()
{
  Material::commitParameters();
  m_pbr = true;
  m_metallic = getParam<float>("metallic", 1.f);
  m_roughness = getParam<float>("roughness", 1.f);
  readCommon("baseColor");
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Material *);
}
