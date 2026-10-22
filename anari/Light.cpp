// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "Light.h"
// std
#include <cmath>

namespace tachyon_anari {

Light::Light(TachyonGlobalState *s) : Object(ANARI_LIGHT, s) {}

Light *Light::createInstance(std::string_view subtype, TachyonGlobalState *s)
{
  if (subtype == "directional")
    return new DirectionalLight(s);
  if (subtype == "point")
    return new PointLight(s);
  return nullptr; // caller creates an UnsupportedLight
}

void DirectionalLight::commitParameters()
{
  m_color = getParam<float3>("color", float3(1.f));
  m_direction = getParam<float3>("direction", float3(0.f, 0.f, -1.f));
  m_irradiance = getParam<float>("irradiance", 1.f);
}

void DirectionalLight::emit(TachyonScene &scene, const mat4 &xfm) const
{
  // ANARI's direction is where the light travels; Tachyon's directional
  // light takes the same convention (verified empirically, see TachyonScene).
  const float4 d = linalg::mul(xfm, float4(m_direction, 0.f));
  float3 dir(d.x, d.y, d.z);
  const float len = linalg::length(dir);
  if (len <= 0.f)
    return;
  dir /= len;
  const float3 c = m_color * m_irradiance;
  scene.addDirectionalLight(&dir.x, &c.x);
}

void PointLight::commitParameters()
{
  m_color = getParam<float3>("color", float3(1.f));
  m_position = getParam<float3>("position", float3(0.f));
  m_intensity = getParam<float>("intensity", 1.f);
  if (hasParam("power", ANARI_FLOAT32)) // total power -> intensity
    m_intensity = getParam<float>("power", 0.f) / (4.f * float(M_PI));
}

void PointLight::emit(TachyonScene &scene, const mat4 &xfm) const
{
  const float4 p = linalg::mul(xfm, float4(m_position, 1.f));
  const float3 pos(p.x, p.y, p.z);
  const float3 c = m_color * m_intensity;
  scene.addPointLight(&pos.x, &c.x);
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Light *);
}
