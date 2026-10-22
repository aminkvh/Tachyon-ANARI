// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "TachyonScene.h"

namespace tachyon_anari {

struct Camera : public Object
{
  Camera(TachyonGlobalState *s);

  static Camera *createInstance(
      std::string_view type, TachyonGlobalState *state);

  void commitParameters() override;

  // Fills the Tachyon camera description. 'frameAspect' is used when the
  // application does not set an explicit aspect ratio.
  virtual void fill(SceneCamera &cam, float frameAspect) const {}

 protected:
  float3 m_pos{0.f, 0.f, 0.f};
  float3 m_dir{0.f, 0.f, 1.f};
  float3 m_up{0.f, 1.f, 0.f};
};

struct PerspectiveCamera : public Camera
{
  PerspectiveCamera(TachyonGlobalState *s) : Camera(s) {}
  void commitParameters() override;
  void fill(SceneCamera &cam, float frameAspect) const override;

 private:
  float m_fovy{anari::radians(60.f)};
  float m_aspect{0.f}; // <= 0: use the frame's aspect ratio
};

struct OrthographicCamera : public Camera
{
  OrthographicCamera(TachyonGlobalState *s) : Camera(s) {}
  void commitParameters() override;
  void fill(SceneCamera &cam, float frameAspect) const override;

 private:
  float m_height{1.f};
  float m_aspect{0.f};
};

using UnsupportedCamera = Unsupported<Camera>;

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Camera *, ANARI_CAMERA);
}
