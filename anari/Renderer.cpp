// Copyright (c) 2026 Tachyon-dev contributors
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
  m_ignoreAmbient = getParam<bool>("ignoreAmbientLighting", true);
  m_eyeLightBlend =
      std::clamp(getParam<float>("eyeLightBlendRatio", 0.5f), 0.f, 1.f);
}

// Tachyon shades a surface as  color * (diffuse * sum(N.L * light) + ambient).
//
// Without application lights the camera headlight stands in for them, mirroring
// the helide device's eye-light blend: ambient = 1 - blend, diffuse = blend.
// With lights, the default (ignoreAmbientLighting) uses a small constant
// ambient term; otherwise the ANARI ambient light is honored as a constant.
void Renderer::shadingWeights(bool lit, float &ambient, float &diffuse) const
{
  if (m_ignoreAmbient) {
    ambient = lit ? 0.3f : 1.f - m_eyeLightBlend;
    diffuse = lit ? 1.f : m_eyeLightBlend;
  } else {
    const float avg =
        (m_ambientColor.x + m_ambientColor.y + m_ambientColor.z) / 3.f;
    ambient = std::clamp(m_ambientRadiance * avg, 0.f, 1.f);
    diffuse = lit ? 1.f : 0.f;
  }
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Renderer *);
}
