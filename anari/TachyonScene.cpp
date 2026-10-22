// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "TachyonScene.h"

extern "C" {
#include "tachyon.h"
}

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace tachyon_anari {

static void initTachyonOnce()
{
  static std::once_flag once;
  std::call_once(once, []() { rt_initialize(nullptr, nullptr); });
}

static apivector vec3(const float *v)
{
  return rt_vector(v[0], v[1], v[2]);
}

static apicolor col3(const float *c)
{
  return rt_color(c[0], c[1], c[2]);
}

TachyonScene::TachyonScene() = default;

TachyonScene::~TachyonScene()
{
  destroy();
}

void TachyonScene::destroy()
{
  if (m_scene)
    rt_deletescene(m_scene);
  m_scene = nullptr;
  m_styles.clear();
  m_headlight = nullptr;
  m_numTriangles = 0;
  m_width = m_height = 0;
  m_lastAspect = -1.f;
  m_lastProjection = -1;
}

void TachyonScene::reset(bool lit, int numThreads)
{
  destroy();
  initTachyonOnce();

  m_scene = rt_newscene();
  m_lit = lit;
  m_transparent = false;

  rt_outputfile(m_scene, ""); // never write image files from the device
  rt_verbose(m_scene, 0);
  rt_image_clamp(m_scene);
  // Keep the default "OFF": interpolated normals are flipped toward the viewer
  // using the winding order, so addTriangle() makes normals agree with it.
  rt_normal_fixup_mode(m_scene, RT_NORMAL_FIXUP_OFF);
  rt_camera_raydepth(m_scene, 8);
  rt_set_numthreads(m_scene, numThreads);

  const char *accel = std::getenv("TACHYON_ANARI_ACCEL");
  if (accel && std::strcmp(accel, "grid") == 0)
    rt_accelstruct(m_scene, RT_ACCEL_GRID);
  else
    rt_accelstruct(m_scene, RT_ACCEL_BVH);
}

int TachyonScene::addStyle(const SurfaceStyle &s)
{
  apitexture tex;
  std::memset(&tex, 0, sizeof(tex));
  tex.texturefunc = RT_TEXTURE_CONSTANT;
  tex.col = col3(s.color);
  tex.shadowcast = m_lit ? 1 : 0;
  tex.ambient = 0.f; // replaced by setShading()
  tex.diffuse = s.diffuse;
  tex.specular = 0.f;
  tex.opacity = s.opacity;
  tex.scale = rt_vector(1.f, 1.f, 1.f);
  tex.uaxs = rt_vector(1.f, 0.f, 0.f);
  tex.vaxs = rt_vector(0.f, 1.f, 0.f);
  tex.waxs = rt_vector(0.f, 0.f, 1.f);

  void *t = rt_texture(m_scene, &tex);
  if (s.phong > 0.f)
    rt_tex_phong(t, s.phong, s.phongExp, s.metal ? 1 : 0);

  if (s.opacity < 1.f)
    m_transparent = true;

  Style st;
  st.diffuse = s.diffuse;
  st.textures.push_back(t);
  m_styles.push_back(std::move(st));
  return int(m_styles.size()) - 1;
}

void TachyonScene::addTriangle(
    int style, const float *p, const float *n, const float *c)
{
  Style &st = m_styles[style];
  void *base = st.textures[0];
  m_numTriangles++;

  // Geometric normal from the winding order
  const float e1[3] = {p[3] - p[0], p[4] - p[1], p[5] - p[2]};
  const float e2[3] = {p[6] - p[0], p[7] - p[1], p[8] - p[2]};
  float gx = e1[1] * e2[2] - e1[2] * e2[1];
  float gy = e1[2] * e2[0] - e1[0] * e2[2];
  float gz = e1[0] * e2[1] - e1[1] * e2[0];

  float fn[9];
  if (c && !n) { // vertex colors need normals: use the flat face normal
    const float len = std::sqrt(gx * gx + gy * gy + gz * gz);
    const float s = len > 0.f ? 1.f / len : 0.f;
    for (int i = 0; i < 3; i++) {
      fn[3 * i + 0] = gx * s;
      fn[3 * i + 1] = gy * s;
      fn[3 * i + 2] = gz * s;
    }
    n = fn;
  } else if (n) {
    // Tachyon orients interpolated normals by the winding order, so the
    // supplied normals must agree with it
    float d = 0.f;
    for (int i = 0; i < 3; i++)
      d += n[3 * i] * gx + n[3 * i + 1] * gy + n[3 * i + 2] * gz;
    if (d < 0.f) {
      for (int i = 0; i < 9; i++)
        fn[i] = -n[i];
      n = fn;
    }
  }

  if (c) {
    // vertex colors live on a per-triangle texture copy
    void *tex = rt_texture_copy_vcstri(m_scene, base);
    st.textures.push_back(tex);
    rt_vcstri3fv(m_scene, tex, p, p + 3, p + 6, n, n + 3, n + 6, c, c + 3, c + 6);
  } else if (n) {
    rt_stri3fv(m_scene, base, p, p + 3, p + 6, n, n + 3, n + 6);
  } else {
    rt_tri3fv(m_scene, base, p, p + 3, p + 6);
  }
}

void TachyonScene::addDirectionalLight(const float *direction, const float *rgb)
{
  apitexture tex;
  std::memset(&tex, 0, sizeof(tex));
  tex.texturefunc = RT_TEXTURE_CONSTANT;
  tex.col = col3(rgb);
  tex.opacity = 1.f;
  void *t = rt_texture(m_scene, &tex);
  rt_directional_light(m_scene, t, vec3(direction)); // same convention as ANARI
}

void TachyonScene::addPointLight(const float *position, const float *rgb)
{
  apitexture tex;
  std::memset(&tex, 0, sizeof(tex));
  tex.texturefunc = RT_TEXTURE_CONSTANT;
  tex.col = col3(rgb);
  tex.opacity = 1.f;
  void *t = rt_texture(m_scene, &tex);
  void *l = rt_light(m_scene, t, vec3(position), 0.f);
  rt_light_attenuation(l, 0.f, 0.f, 1.f); // inverse square falloff
}

void TachyonScene::finishBuild()
{
  if (m_lit)
    return;
  // The camera headlight can never be shadowed by anything the camera sees,
  // so opaque scenes use Tachyon's medium shader, which skips shadow rays.
  if (!m_transparent)
    rt_shadermode(m_scene, RT_SHADER_MEDIUM);
  const float white[3] = {1.f, 1.f, 1.f};
  const float dir[3] = {0.f, 0.f, 1.f}; // re-aimed by setCamera()
  apitexture tex;
  std::memset(&tex, 0, sizeof(tex));
  tex.texturefunc = RT_TEXTURE_CONSTANT;
  tex.col = col3(white);
  tex.opacity = 1.f;
  void *t = rt_texture(m_scene, &tex);
  m_headlight = rt_directional_light(m_scene, t, vec3(dir));
}

void TachyonScene::setShading(float ambient, float diffuseScale)
{
  for (auto &st : m_styles) {
    for (void *vt : st.textures) {
      rt_tex_shading(vt, ambient, diffuseScale * st.diffuse);
    }
  }
}

void TachyonScene::setDepthOutput(float *depthBuffer)
{
  rt_depthbuffer(m_scene, depthBuffer);
}

void TachyonScene::setBackground(const float *rgb)
{
  rt_background(m_scene, col3(rgb));
}

void TachyonScene::setSamples(int pixelSamples)
{
  rt_aa_maxsamples(m_scene, pixelSamples > 1 ? pixelSamples - 1 : 0);
}

void TachyonScene::setOutput(int width, int height, float *rgbBuffer)
{
  if (width != m_width || height != m_height || rgbBuffer != m_buffer) {
    // these mark the scene for re-preparation, so only call on a real change
    rt_resolution(m_scene, width, height);
    rt_rawimage_rgb96f(m_scene, rgbBuffer);
    m_width = width;
    m_height = height;
    m_buffer = rgbBuffer;
    m_lastAspect = -1.f;
  }
}

void TachyonScene::setCamera(const SceneCamera &cam)
{
  const int projection =
      cam.orthographic ? RT_PROJECTION_ORTHOGRAPHIC : RT_PROJECTION_PERSPECTIVE;
  if (projection != m_lastProjection) {
    rt_camera_projection(m_scene, projection);
    m_lastProjection = projection;
  }

  const float frameAspect = float(m_width) / float(m_height > 0 ? m_height : 1);
  const float aspect = cam.aspect > 0.f ? cam.aspect : frameAspect;
  const float aspectRatio = frameAspect / aspect;
  if (aspectRatio != m_lastAspect) { // setting it re-prepares the scene
    rt_aspectratio(m_scene, aspectRatio);
    m_lastAspect = aspectRatio;
  }

  // Tachyon's image plane is 1/zoom tall at distance 1.
  const float zoom = cam.orthographic
      ? 1.f / (cam.height > 0.f ? cam.height : 1.f)
      : 0.5f / std::tan(0.5f * cam.fovy);
  rt_camera_zoom(m_scene, zoom); // also resets the frustum to automatic

  const float *r = cam.region;
  if (r[0] != 0.f || r[1] != 0.f || r[2] != 1.f || r[3] != 1.f) {
    // image plane is (aspect / zoom) x (1 / zoom); Tachyon's vertical axis is
    // flipped relative to ANARI's (see below)
    const float pw = aspect / zoom, ph = 1.f / zoom;
    rt_camera_frustum(m_scene,
        (r[0] - 0.5f) * pw,
        (r[2] - 0.5f) * pw,
        (0.5f - r[3]) * ph,
        (0.5f - r[1]) * ph);
  }

  // Tachyon's camera basis is left-handed (right = up x view) while ANARI's
  // is right-handed (right = view x up). Negating 'up' flips Tachyon's right
  // vector to ANARI's; the vertical flip this causes is undone by the caller
  // reading the rows bottom-up (see Frame).
  const float negUp[3] = {-cam.up[0], -cam.up[1], -cam.up[2]};
  rt_camera_position3fv(m_scene, cam.position, cam.direction, negUp);

  if (m_headlight) {
    // headlight sits at the camera: it travels along the view direction
    rt_directional_light_set_direction(m_headlight, vec3(cam.direction));
  }
}

void TachyonScene::render()
{
  rt_renderscene(m_scene);
}

} // namespace tachyon_anari
