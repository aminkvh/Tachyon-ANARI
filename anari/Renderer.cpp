// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#include "Renderer.h"
// std
#include <algorithm>

namespace tachyon_anari {

Renderer::Renderer(TachyonGlobalState *s) : Object(ANARI_RENDERER, s) {}

void Renderer::commitParameters()
{
  m_background = getParam<float4>("background", float4(0.f, 0.f, 0.f, 1.f));
  m_pixelSamples = std::max(1, getParam<int32_t>("pixelSamples", 1));
  m_ambientRadiance = getParam<float>("ambientRadiance", 1.f);
  m_ambientColor = getParam<float3>("ambientColor", float3(1.f));
  const std::string mode = getParamString("mode", "default");
  m_eyeLight = mode == "eyeLight";
  // like helide, the preview mode defaults to ignoring ambientRadiance
  m_ignoreAmbient = getParam<bool>("ignoreAmbientLighting", m_eyeLight);
  m_eyeLightBlend =
      std::clamp(getParam<float>("eyeLightBlendRatio", 0.5f), 0.f, 1.f);
}

// Tachyon shades a surface as  color * (diffuse * sum(N.L * light) + ambient).
//
// With application lights: ambient = ambientRadiance * ambientColor (a constant
// ambient light, as in the ANARI spec) and the lights add their Lambertian
// diffuse term (see Light.cpp for the 1/pi).
// Without lights a camera headlight stands in for them, mirroring helide's
// eye-light blend: ambient = (1 - blend), diffuse = blend, both scaled by the
// ambient light level. ignoreAmbientLighting replaces that level by 1 and the
// ambient term in lit scenes by a fixed 0.3.
//
// mode "eyeLight" (helide's shading model, a cheap preview): lights are never
// used, no shadow rays are cast. A surface gets
//   color * ambientRadiance * ((1 - blend) + blend * |N.V|)
// where V points to the camera; ambientColor is not used (as in helide) and
// ignoreAmbientLighting (default true in this mode) forces ambientRadiance 1.
void Renderer::shadingWeights(bool lit, float &ambient, float &diffuse) const
{
  if (m_eyeLight) {
    const float level = m_ignoreAmbient ? 1.f : std::max(0.f, m_ambientRadiance);
    ambient = level * (1.f - m_eyeLightBlend);
    diffuse = level * m_eyeLightBlend;
    return;
  }
  const float avg =
      (m_ambientColor.x + m_ambientColor.y + m_ambientColor.z) / 3.f;
  const float level = m_ignoreAmbient ? 1.f : std::max(0.f, m_ambientRadiance * avg);
  if (lit) {
    ambient = m_ignoreAmbient ? 0.3f : level;
    diffuse = 1.f;
  } else {
    ambient = level * (1.f - m_eyeLightBlend);
    diffuse = level * m_eyeLightBlend;
  }
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Renderer *);
}
