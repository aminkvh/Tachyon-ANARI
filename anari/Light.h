// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "TachyonScene.h"

namespace tachyon_anari {

struct Light : public Object
{
  Light(TachyonGlobalState *s);

  static Light *createInstance(std::string_view subtype, TachyonGlobalState *s);

  bool affectsScene() const override { return true; }

  // Adds the light to the Tachyon scene, transformed by 'xfm'
  virtual void emit(TachyonScene &, const mat4 &) const {}

 protected:
  float3 m_color{1.f, 1.f, 1.f};
};

struct DirectionalLight : public Light
{
  DirectionalLight(TachyonGlobalState *s) : Light(s) {}
  void commitParameters() override;
  void emit(TachyonScene &scene, const mat4 &xfm) const override;

 private:
  float3 m_direction{0.f, 0.f, -1.f}; // direction the light travels
  float m_irradiance{1.f};
};

struct PointLight : public Light
{
  PointLight(TachyonGlobalState *s) : Light(s) {}
  void commitParameters() override;
  void emit(TachyonScene &scene, const mat4 &xfm) const override;

 private:
  float3 m_position{0.f, 0.f, 0.f};
  float m_intensity{1.f};
};

using UnsupportedLight = Unsupported<Light>;

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Light *, ANARI_LIGHT);
}
