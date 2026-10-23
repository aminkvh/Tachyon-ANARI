// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#include "Camera.h"

namespace tachyon_anari {

Camera::Camera(TachyonGlobalState *s) : Object(ANARI_CAMERA, s) {}

Camera *Camera::createInstance(std::string_view type, TachyonGlobalState *s)
{
  if (type == "perspective")
    return new PerspectiveCamera(s);
  if (type == "orthographic")
    return new OrthographicCamera(s);
  return nullptr; // caller creates an UnsupportedCamera
}

void Camera::commitParameters()
{
  m_pos = getParam<float3>("position", float3(0.f));
  m_dir = linalg::normalize(getParam<float3>("direction", float3(0.f, 0.f, 1.f)));
  m_up = linalg::normalize(getParam<float3>("up", float3(0.f, 1.f, 0.f)));
  m_region[0] = m_region[1] = 0.f;
  m_region[2] = m_region[3] = 1.f;
  getParam("imageRegion", ANARI_FLOAT32_BOX2, m_region);
}

static void fillBasics(SceneCamera &c,
    const float3 &pos,
    const float3 &dir,
    const float3 &up)
{
  for (int i = 0; i < 3; i++) {
    c.position[i] = pos[i];
    c.direction[i] = dir[i];
    c.up[i] = up[i];
  }
}

void Camera::fillRegion(SceneCamera &c) const
{
  for (int i = 0; i < 4; i++)
    c.region[i] = m_region[i];
}

void PerspectiveCamera::commitParameters()
{
  Camera::commitParameters();
  m_fovy = getParam<float>("fovy", anari::radians(60.f));
  m_aspect = hasParam("aspect", ANARI_FLOAT32) ? getParam<float>("aspect", 0.f)
                                                : 0.f;
}

void PerspectiveCamera::fill(SceneCamera &c, float) const
{
  fillBasics(c, m_pos, m_dir, m_up);
  c.orthographic = false;
  c.fovy = m_fovy;
  c.aspect = m_aspect;
  fillRegion(c);
}

void OrthographicCamera::commitParameters()
{
  Camera::commitParameters();
  m_height = getParam<float>("height", 1.f);
  m_aspect = hasParam("aspect", ANARI_FLOAT32) ? getParam<float>("aspect", 0.f)
                                                : 0.f;
}

void OrthographicCamera::fill(SceneCamera &c, float) const
{
  fillBasics(c, m_pos, m_dir, m_up);
  c.orthographic = true;
  c.height = m_height;
  c.aspect = m_aspect;
  fillRegion(c);
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Camera *);
}
