// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"

namespace tachyon_anari {

struct Renderer : public Object
{
  Renderer(TachyonGlobalState *s);

  void commitParameters() override;

  const float4 &background() const { return m_background; }
  int pixelSamples() const { return m_pixelSamples; }
  // "eyeLight" preview mode: application lights are ignored and surfaces are
  // shaded with the camera headlight blended with ambient (see Renderer.cpp).
  bool eyeLight() const { return m_eyeLight; }

  // Weights used for the surface shading, see Renderer.cpp.
  // 'lit': the world contains application lights.
  void shadingWeights(bool lit, float &ambient, float &diffuse) const;

 private:
  float4 m_background{0.f, 0.f, 0.f, 1.f};
  int m_pixelSamples{1};
  float m_ambientRadiance{1.f};
  float3 m_ambientColor{1.f, 1.f, 1.f};
  bool m_ignoreAmbient{false};
  float m_eyeLightBlend{0.5f};
  bool m_eyeLight{false};
};

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Renderer *, ANARI_RENDERER);
}
