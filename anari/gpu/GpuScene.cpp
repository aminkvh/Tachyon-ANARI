// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "GpuScene.h"

#include "../Backends.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace tachyon_anari {
namespace gpu {

namespace {

enum : unsigned
{
  WARN_SAMPLER = 1,
  WARN_SAMPLES = 2
};


uint64_t fnv(uint64_t h, const void *data, size_t bytes)
{
  const unsigned char *p = static_cast<const unsigned char *>(data);
  for (size_t i = 0; i < bytes; i++) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

void normalize3(const float *a, float *o)
{
  const float l = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  const float s = l > 0.f ? 1.f / l : 0.f;
  o[0] = a[0] * s;
  o[1] = a[1] * s;
  o[2] = a[2] * s;
}

float bitsToFloat(uint32_t u)
{
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// The CPU backend's normal handling (TachyonScene::addTriangle): flat faces
// use the geometric normal, interpolated normals are made to agree with the
// winding.
void triangleNormals(const float *p, const float *n, bool forceFlat, float *out)
{
  const float e1[3] = {p[3] - p[0], p[4] - p[1], p[5] - p[2]};
  const float e2[3] = {p[6] - p[0], p[7] - p[1], p[8] - p[2]};
  const float gx = e1[1] * e2[2] - e1[2] * e2[1];
  const float gy = e1[2] * e2[0] - e1[0] * e2[2];
  const float gz = e1[0] * e2[1] - e1[1] * e2[0];
  if (!n || forceFlat) {
    const float len = std::sqrt(gx * gx + gy * gy + gz * gz);
    const float s = len > 0.f ? 1.f / len : 0.f;
    for (int i = 0; i < 3; i++) {
      out[3 * i + 0] = gx * s;
      out[3 * i + 1] = gy * s;
      out[3 * i + 2] = gz * s;
    }
    return;
  }
  float d = 0.f;
  for (int i = 0; i < 3; i++)
    d += n[3 * i] * gx + n[3 * i + 1] * gy + n[3 * i + 2] * gz;
  const float sign = d < 0.f ? -1.f : 1.f;
  for (int i = 0; i < 9; i++)
    out[i] = sign * n[i];
}

} // namespace

GpuSceneBase::GpuSceneBase()
{
  m_debug = debugEnabled();
  m_protos.emplace_back();
}

void GpuSceneBase::warnOnce(unsigned id, const char *text)
{
  if (m_warned & id)
    return;
  m_warned |= id;
  m_warnings.emplace_back(text);
}

std::vector<std::string> GpuSceneBase::takeWarnings()
{
  std::vector<std::string> w;
  w.swap(m_warnings);
  return w;
}

// Scene description ///////////////////////////////////////////////////////////

// The GPU scene of the previous build stays alive: finishBuild() decides
// whether it can be refitted.
void GpuSceneBase::reset(bool lit, int)
{
  m_styles.clear();
  m_protos.clear();
  m_protos.emplace_back();
  m_cur = 0;
  m_placements.clear();
  m_samplerIds.clear();
  m_samplers.clear();
  m_lights.clear();
  m_numSpheres = m_numTriangles = m_numCylinders = 0;
  m_lit = lit;
  m_transparent = false;
  m_unsupported = false;
  m_buildId++;
}

int GpuSceneBase::addStyle(const SurfaceStyle &style)
{
  m_styles.push_back(style);
  if (style.opacity < 1.f)
    m_transparent = true;
  return int(m_styles.size()) - 1;
}

void GpuSceneBase::addDirectionalLight(const float *direction, const float *rgb)
{
  Light l{};
  normalize3(direction, l.v);
  for (int i = 0; i < 3; i++) {
    l.v[i] = -l.v[i]; // towards the light
    l.color[i] = rgb[i];
  }
  l.type = 0;
  m_lights.push_back(l);
}

void GpuSceneBase::addPointLight(const float *position, const float *rgb)
{
  Light l{};
  for (int i = 0; i < 3; i++) {
    l.v[i] = position[i];
    l.color[i] = rgb[i];
  }
  l.type = 1;
  m_lights.push_back(l);
}

int GpuSceneBase::beginPrototype()
{
  m_protos.emplace_back();
  m_cur = m_protos.size() - 1;
  return int(m_cur);
}

void GpuSceneBase::endPrototype()
{
  m_cur = 0;
}

void GpuSceneBase::addInstance(int prototype, const float *x)
{
  if (prototype <= 0 || size_t(prototype) >= m_protos.size())
    return;
  Placement pl;
  pl.proto = prototype;
  for (int r = 0; r < 3; r++)
    for (int c = 0; c < 4; c++)
      pl.m[4 * r + c] = x[4 * c + r]; // column major 4x4 to row major 3x4
  m_placements.push_back(pl);
}

// Triangles ///////////////////////////////////////////////////////////////////

void GpuSceneBase::addTriangle(
    int style, const float *p, const float *n, const float *c)
{
  Proto &pr = m_protos[m_cur];
  float nn[9];
  triangleNormals(p, n, c && !n, nn);
  uint32_t flags = 0;
  uint32_t aux = uint32_t(pr.triAux.size());
  if (c) {
    flags |= 1;
    pr.triAux.insert(pr.triAux.end(), c, c + 9);
  }
  pr.triPos.insert(pr.triPos.end(), p, p + 9);
  pr.triShade.insert(pr.triShade.end(), nn, nn + 9);
  pr.triShade.push_back(bitsToFloat(uint32_t(style)));
  pr.triShade.push_back(bitsToFloat(flags));
  pr.triShade.push_back(bitsToFloat(aux));
  m_numTriangles++;
}

void GpuSceneBase::addSampledTriangle(int style,
    const float *p,
    const float *n,
    const float *vertexAttr,
    AttrSampleFn,
    const void *user,
    const SamplerDesc *desc)
{
  if (!desc) {
    m_unsupported = true;
    warnOnce(WARN_SAMPLER,
        "a sampler cannot be evaluated on the GPU, its surface uses the "
        "material color");
    addTriangle(style, p, n, nullptr);
    return;
  }
  Proto &pr = m_protos[m_cur];
  auto it = m_samplerIds.find(user);
  int sid;
  if (it == m_samplerIds.end()) {
    sid = int(m_samplers.size());
    m_samplers.push_back(*desc);
    m_samplerIds.emplace(user, sid);
  } else {
    sid = it->second;
  }
  float nn[9];
  triangleNormals(p, n, false, nn);
  const uint32_t flags = 2u | (uint32_t(sid) << 8);
  const uint32_t aux = uint32_t(pr.triAux.size());
  pr.triAux.insert(pr.triAux.end(), vertexAttr, vertexAttr + 12);
  pr.triPos.insert(pr.triPos.end(), p, p + 9);
  pr.triShade.insert(pr.triShade.end(), nn, nn + 9);
  pr.triShade.push_back(bitsToFloat(uint32_t(style)));
  pr.triShade.push_back(bitsToFloat(flags));
  pr.triShade.push_back(bitsToFloat(aux));
  m_numTriangles++;
}

// Spheres /////////////////////////////////////////////////////////////////////

// Colors / styles are only stored per primitive when they are not all the
// same; these fill the arrays from the runs added so far.
void GpuSceneBase::ensureColors(Proto &p, bool sphere)
{
  bool &active = sphere ? p.sphereColorsActive : p.cylColorsActive;
  if (active)
    return;
  active = true;
  const std::vector<Run> &runs = sphere ? p.sphereRuns : p.cylRuns;
  std::vector<float> &col = sphere ? p.sphereColors : p.cylColors;
  for (const auto &run : runs) {
    const float *c = m_styles[run.style].color;
    for (size_t i = 0; i < run.count; i++) {
      col.insert(col.end(), c, c + 3);
      if (!sphere)
        col.insert(col.end(), c, c + 3);
    }
  }
}

void GpuSceneBase::ensureStyles(Proto &p, bool sphere)
{
  bool &active = sphere ? p.sphereStylesActive : p.cylStylesActive;
  if (active)
    return;
  active = true;
  const std::vector<Run> &runs = sphere ? p.sphereRuns : p.cylRuns;
  std::vector<uint32_t> &sty = sphere ? p.sphereStyles : p.cylStyles;
  for (const auto &run : runs)
    sty.insert(sty.end(), run.count, uint32_t(run.style));
}

void GpuSceneBase::addSphere(
    int style, const float *center, float radius, const float *rgb)
{
  addSpheres(style, 1, center, &radius, radius, rgb);
}

void GpuSceneBase::addSpheres(int style,
    size_t count,
    const float *center,
    const float *radius,
    float uniformRadius,
    const float *rgb)
{
  if (count == 0)
    return;
  Proto &p = m_protos[m_cur];
  const float *styleColor = m_styles[style].color;
  if (p.spheres.empty())
    p.sphereStyle0 = style;
  else if (!p.sphereStylesActive && style != p.sphereStyle0)
    ensureStyles(p, true); // spheres of different styles
  if (rgb)
    ensureColors(p, true);

  p.spheres.reserve(p.spheres.size() + 4 * count);
  if (p.sphereColorsActive)
    p.sphereColors.reserve(p.sphereColors.size() + 3 * count);
  size_t kept = 0;
  for (size_t i = 0; i < count; i++) {
    const float r = radius ? radius[i] : uniformRadius;
    if (!(r > 0.f))
      continue;
    p.spheres.push_back(center[3 * i]);
    p.spheres.push_back(center[3 * i + 1]);
    p.spheres.push_back(center[3 * i + 2]);
    p.spheres.push_back(r);
    if (p.sphereColorsActive) {
      const float *c = rgb ? rgb + 3 * i : styleColor;
      p.sphereColors.insert(p.sphereColors.end(), c, c + 3);
    }
    kept++;
  }
  if (p.sphereStylesActive)
    p.sphereStyles.insert(p.sphereStyles.end(), kept, uint32_t(style));
  p.sphereRuns.push_back({kept, style});
  m_numSpheres += kept;
}

// Cylinders ///////////////////////////////////////////////////////////////////

void GpuSceneBase::addCylinder(int style,
    const float *p0,
    const float *p1,
    float radius,
    const float *rgb0,
    const float *rgb1,
    int caps)
{
  const float ax = p1[0] - p0[0], ay = p1[1] - p0[1], az = p1[2] - p0[2];
  if (!(ax * ax + ay * ay + az * az > 0.f))
    return; // degenerate
  Proto &p = m_protos[m_cur];
  if (p.cyls.empty())
    p.cylStyle0 = style;
  else if (!p.cylStylesActive && style != p.cylStyle0)
    ensureStyles(p, false);
  // CPU rule: split into two colored halves only when both colors are given
  // and differ; otherwise the first color (or the style color) is used
  const float *c0 = rgb0 ? rgb0 : (rgb1 ? rgb1 : nullptr);
  const bool split = rgb0 && rgb1
      && (rgb0[0] != rgb1[0] || rgb0[1] != rgb1[1] || rgb0[2] != rgb1[2]);
  if (c0)
    ensureColors(p, false);

  p.cyls.insert(p.cyls.end(), {p0[0], p0[1], p0[2], radius});
  p.cyls.insert(
      p.cyls.end(), {p1[0], p1[1], p1[2], bitsToFloat(uint32_t(caps & 3))});
  if (p.cylColorsActive) {
    const float *a = c0 ? c0 : m_styles[style].color;
    const float *b = split ? rgb1 : a;
    p.cylColors.insert(p.cylColors.end(), a, a + 3);
    p.cylColors.insert(p.cylColors.end(), b, b + 3);
  }
  if (p.cylStylesActive)
    p.cylStyles.push_back(uint32_t(style));
  p.cylRuns.push_back({1, style});
  m_numCylinders++;
}

// Brings the host description into its final layout: the arrays of all
// prototypes one after the other, one BLAS "slice" per prototype and type, one
// instance entry per placement and slice.
GpuSceneBase::Gathered GpuSceneBase::gather()
{
  Gathered g;
  const size_t np = m_protos.size();

  // do all prototypes keep per primitive colors / styles?
  bool colS = false, styS = false, colC = false, styC = false;
  int s0 = -1, c0 = -1;
  for (const Proto &p : m_protos) {
    if (!p.spheres.empty()) {
      colS |= p.sphereColorsActive;
      if (s0 < 0)
        s0 = p.sphereStyle0;
      styS |= p.sphereStylesActive || p.sphereStyle0 != s0;
    }
    if (!p.cyls.empty()) {
      colC |= p.cylColorsActive;
      if (c0 < 0)
        c0 = p.cylStyle0;
      styC |= p.cylStylesActive || p.cylStyle0 != c0;
    }
  }
  g.styleUniform[0] = uint32_t(std::max(s0, 0));
  g.styleUniform[1] = uint32_t(std::max(c0, 0));
  if (colS)
    g.flags |= F_SPHERE_COLORS;
  if (styS)
    g.flags |= F_SPHERE_STYLES;
  if (colC)
    g.flags |= F_CYL_COLORS;
  if (styC)
    g.flags |= F_CYL_STYLES;

  std::vector<std::vector<int>> sliceOf(np, std::vector<int>(3, -1));
  uint32_t baseS = 0, baseC = 0, baseT = 0;
  uint32_t auxBase = 0;
  for (size_t pi = 0; pi < np; pi++) {
    Proto &p = m_protos[pi];
    const uint32_t ns = uint32_t(p.spheres.size() / 4);
    const uint32_t nc = uint32_t(p.cyls.size() / 8);
    const uint32_t nt = uint32_t(p.triPos.size() / 9);
    if (colS && ns)
      ensureColors(p, true);
    if (styS && ns)
      ensureStyles(p, true);
    if (colC && nc)
      ensureColors(p, false);
    if (styC && nc)
      ensureStyles(p, false);
    const bool single = np == 1;
    auto append = [&](auto &dst, auto &src) {
      if (single)
        dst = std::move(src);
      else
        dst.insert(dst.end(), src.begin(), src.end());
    };
    if (ns) {
      sliceOf[pi][0] = int(g.slices.size());
      g.slices.push_back({int(pi), 0, baseS, ns});
      baseS += ns;
      append(g.spheres, p.spheres);
      if (colS)
        append(g.sphereColors, p.sphereColors);
      if (styS)
        append(g.sphereStyles, p.sphereStyles);
    }
    if (nc) {
      sliceOf[pi][1] = int(g.slices.size());
      g.slices.push_back({int(pi), 1, baseC, nc});
      baseC += nc;
      append(g.cyls, p.cyls);
      if (colC)
        append(g.cylColors, p.cylColors);
      if (styC)
        append(g.cylStyles, p.cylStyles);
    }
    if (nt) {
      sliceOf[pi][2] = int(g.slices.size());
      g.slices.push_back({int(pi), 2, baseT, nt});
      baseT += nt;
      const size_t first = g.triShade.size();
      append(g.triPos, p.triPos);
      append(g.triShade, p.triShade);
      if (auxBase && !single) {
        // attribute offsets are relative to the prototype's own array
        for (uint32_t t = 0; t < nt; t++) {
          float *rec = &g.triShade[first + 12 * size_t(t)];
          uint32_t flags, aux;
          std::memcpy(&flags, rec + 10, 4);
          std::memcpy(&aux, rec + 11, 4);
          if (flags & 3u) {
            aux += auxBase;
            std::memcpy(rec + 11, &aux, 4);
          }
        }
      }
      auxBase += uint32_t(p.triAux.size());
      append(g.triAux, p.triAux);
    }
  }
  g.nSph = baseS;
  g.nCyl = baseC;
  g.nTri = baseT;

  // style table: (color, opacity), (diffuse, phong, phong exponent, metal)
  for (const auto &s : m_styles) {
    g.styles.insert(g.styles.end(), {s.color[0], s.color[1], s.color[2], s.opacity});
    g.styles.insert(g.styles.end(), {s.diffuse, s.phong, s.phongExp, s.metal ? 1.f : 0.f});
  }
  if (g.styles.empty())
    g.styles.assign(8, 1.f);

  // sampler table + texels
  for (const auto &d : m_samplers) {
    const uint32_t texelBase = uint32_t(g.texels.size() / 4);
    const uint32_t hdr0[4] = {uint32_t(d.dims),
        uint32_t(d.size[0]),
        uint32_t(d.size[1]),
        uint32_t(d.size[2])};
    const uint32_t hdr1[4] = {uint32_t(d.wrap[0]),
        uint32_t(d.wrap[1]),
        uint32_t(d.wrap[2]),
        d.linear ? 1u : 0u};
    const uint32_t hdr2[4] = {texelBase, 0, 0, 0};
    for (int k = 0; k < 4; k++)
      g.samplerTable.push_back(bitsToFloat(hdr0[k]));
    for (int k = 0; k < 4; k++)
      g.samplerTable.push_back(bitsToFloat(hdr1[k]));
    for (int k = 0; k < 4; k++)
      g.samplerTable.push_back(bitsToFloat(hdr2[k]));
    g.samplerTable.insert(g.samplerTable.end(), d.inTransform, d.inTransform + 16);
    g.samplerTable.insert(g.samplerTable.end(), d.inOffset, d.inOffset + 4);
    g.samplerTable.insert(g.samplerTable.end(), d.outTransform, d.outTransform + 16);
    g.samplerTable.insert(g.samplerTable.end(), d.outOffset, d.outOffset + 4);
    g.texels.insert(g.texels.end(), d.texels.begin(), d.texels.end());
  }

  // instances: the scene itself once, then every placement of a prototype
  auto addEntries = [&](size_t proto, const float *m) {
    static const float ident[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
    for (int t = 0; t < 3; t++) {
      const int sl = sliceOf[proto][t];
      if (sl < 0)
        continue;
      InstanceEntry e;
      e.slice = sl;
      e.flags = 0;
      if (!m) {
        std::memcpy(e.m, ident, sizeof(ident));
        e.flags |= 1;
      } else {
        std::memcpy(e.m, m, sizeof(e.m));
        if (std::memcmp(m, ident, sizeof(ident)) == 0)
          e.flags |= 1;
        const float det = m[0] * (m[5] * m[10] - m[6] * m[9])
            - m[1] * (m[4] * m[10] - m[6] * m[8])
            + m[2] * (m[4] * m[9] - m[5] * m[8]);
        if (det < 0.f)
          e.flags |= 2;
      }
      g.instances.push_back(e);
    }
  };
  addEntries(0, nullptr);
  for (const Placement &pl : m_placements)
    addEntries(size_t(pl.proto), pl.m);

  // instance records for the shader
  for (const InstanceEntry &e : g.instances) {
    const Slice &sl = g.slices[size_t(e.slice)];
    g.instTable.push_back(bitsToFloat(uint32_t(sl.type)));
    g.instTable.push_back(bitsToFloat(sl.base));
    g.instTable.push_back(bitsToFloat(e.flags));
    g.instTable.push_back(0.f);
    g.instTable.insert(g.instTable.end(), e.m, e.m + 12);
  }

  // what a refit may not change
  for (const Slice &s : g.slices) {
    g.sig.layout.push_back(uint32_t(s.proto));
    g.sig.layout.push_back(uint32_t(s.type));
    g.sig.layout.push_back(s.count);
  }
  for (const InstanceEntry &e : g.instances)
    g.sig.layout.push_back(uint32_t(e.slice));
  g.sig.styles = g.styles;
  g.sig.flags = g.flags;
  g.sig.aux = g.triAux.size();
  g.sig.texels = g.texels.size();
  g.sig.samplers = m_samplers.size();
  return g;
}

// Frame ///////////////////////////////////////////////////////////////////////

void GpuSceneBase::setShading(float ambient, float diffuseScale)
{
  m_ambient = ambient;
  m_diffuse = diffuseScale;
}

void GpuSceneBase::setBackground(const float *rgb)
{
  std::memcpy(m_bg, rgb, 3 * sizeof(float));
}

void GpuSceneBase::setSamples(int pixelSamples)
{
  m_samples = std::max(1, pixelSamples);
}

void GpuSceneBase::setCamera(const SceneCamera &cam)
{
  m_cam = cam;
}

void GpuSceneBase::setOutput(int width, int height, float *)
{
  m_width = width;
  m_height = height;
}

void GpuSceneBase::setDepthOutput(float *) {}

bool GpuSceneBase::setDirectOutput(
    OutputFormat format, void *color, float *depth, float bgAlpha)
{
  m_format = color ? format : OutputFormat::None;
  m_colorDst = color;
  m_depthDst = depth;
  m_bg[3] = bgAlpha;
  return true;
}

void GpuSceneBase::fillDirectBackground()
{
  const size_t np = size_t(m_width) * size_t(m_height);
  if (m_colorDst) {
    const float a = std::min(std::max(m_bg[3], 0.f), 1.f);
    const uint32_t ua = uint32_t(255.f * a);
    float c[3];
    for (int i = 0; i < 3; i++)
      c[i] = std::min(std::max(m_bg[i], 0.f), 1.f);
    if (m_format == OutputFormat::Float32) {
      float *dst = static_cast<float *>(m_colorDst);
      for (size_t i = 0; i < np; i++) {
        dst[4 * i] = c[0];
        dst[4 * i + 1] = c[1];
        dst[4 * i + 2] = c[2];
        dst[4 * i + 3] = m_bg[3];
      }
    } else if (m_format != OutputFormat::None) {
      uint32_t v[3];
      for (int i = 0; i < 3; i++) {
        float f = c[i];
        if (m_format == OutputFormat::UFixed8Srgb)
          f = std::pow(f, 1.f / 2.2f);
        v[i] = uint32_t(255.f * std::min(std::max(f, 0.f), 1.f));
      }
      const uint32_t px = v[0] | (v[1] << 8) | (v[2] << 16) | (ua << 24);
      uint32_t *dst = static_cast<uint32_t *>(m_colorDst);
      std::fill(dst, dst + np, px);
    }
  }
  if (m_depthDst)
    std::fill(m_depthDst, m_depthDst + np, std::numeric_limits<float>::infinity());
}

// Per frame constants: camera, lights, flags and the progressive refinement
// state (identical consecutive frames add samples to an accumulation buffer)
GpuSceneBase::FrameSetup GpuSceneBase::prepareFrame()
{
  FrameSetup out;
  const size_t np = size_t(m_width) * size_t(m_height);
  const bool wantDepth = m_depthDst != nullptr;
  const OutputFormat fmt = m_format;
  const size_t pixBytes = fmt == OutputFormat::None ? 16
      : fmt == OutputFormat::Float32                ? 16 * np
                                                    : 4 * np;

  // Per frame constants ------------------------------------------------------
  const SceneCamera &c = m_cam;
  float f[3], r[3], u[3], t[3];
  normalize3(c.direction, f);
  const float rr[3] = {f[1] * c.up[2] - f[2] * c.up[1],
      f[2] * c.up[0] - f[0] * c.up[2],
      f[0] * c.up[1] - f[1] * c.up[0]};
  normalize3(rr, r);
  t[0] = r[1] * f[2] - r[2] * f[1];
  t[1] = r[2] * f[0] - r[0] * f[2];
  t[2] = r[0] * f[1] - r[1] * f[0];
  normalize3(t, u);
  const float frameAspect = float(m_width) / float(m_height);
  const float aspect = c.aspect > 0.f ? c.aspect : frameAspect;
  const float ph = c.orthographic ? (c.height > 0.f ? c.height : 1.f)
                                  : 2.f * std::tan(0.5f * c.fovy);
  const float pw = aspect * ph;
  const float *reg = c.region;

  FrameData &fd = out.fd;
  fd = FrameData{};
  for (int i = 0; i < 3; i++) {
    fd.posOrtho[i] = c.position[i];
    fd.fwdU0[i] = f[i];
    fd.rightDu[i] = r[i];
    fd.upV0[i] = u[i];
    fd.bg[i] = m_bg[i];
  }
  fd.bg[3] = m_bg[3];
  fd.posOrtho[3] = c.orthographic ? 1.f : 0.f;
  fd.fwdU0[3] = (reg[0] - 0.5f) * pw;
  fd.rightDu[3] = (reg[2] - reg[0]) * pw / float(m_width);
  fd.upV0[3] = (reg[3] - 0.5f) * ph;
  fd.shade[0] = m_ambient;
  fd.shade[1] = m_diffuse;
  fd.shade[2] = -(reg[3] - reg[1]) * ph / float(m_height);
  // The CPU backend (verified against helide with the CTS) shoots each ray
  // through the lower-left corner of its pixel in ANARI's orientation, i.e.
  // half a pixel left of and half a pixel below the pixel center the shader
  // uses. Measured: this reproduces the CPU image to 91 dB.
  fd.fwdU0[3] -= 0.5f * fd.rightDu[3];
  fd.upV0[3] += 0.5f * fd.shade[2];
  fd.size[0] = uint32_t(m_width);
  fd.size[1] = uint32_t(m_height);
  fd.flags[0] = m_sceneFlags | ((m_lit || m_transparent) ? F_SHADOWS : 0u)
      | (m_transparent ? F_TRANSPARENT : 0u)
      | (wantDepth ? F_DEPTH : 0u) | (m_bg[3] < 1.f ? F_BG_ALPHA : 0u);
  fd.flags[1] = uint32_t(fmt);
  fd.flags[2] = 8; // Tachyon ray depth used by the CPU backend
  fd.styles[0] = m_styleUniform[0];
  fd.styles[1] = m_styleUniform[1];

  // Lights: the application's, or a headlight along the view direction
  std::vector<float> &lightData = out.lights;
  if (m_lit) {
    for (const auto &l : m_lights) {
      lightData.insert(lightData.end(), {l.v[0], l.v[1], l.v[2], float(l.type)});
      lightData.insert(lightData.end(), {l.color[0], l.color[1], l.color[2], 0.f});
    }
  } else {
    lightData = {-f[0], -f[1], -f[2], 0.f, 1.f, 1.f, 1.f, 0.f};
  }
  fd.counts[3] = uint32_t(lightData.size() / 8);

  // Progressive refinement: identical consecutive frames add samples
  uint64_t h = 1469598103934665603ull;
  h = fnv(h, &fd, sizeof(fd));
  h = fnv(h, lightData.data(), 4 * lightData.size());
  h = fnv(h, &m_buildId, sizeof(m_buildId));
  h = fnv(h, &m_samples, sizeof(m_samples));
  h = fnv(h, &m_colorDst, sizeof(m_colorDst));
  if (h == m_lastFrameHash)
    m_sameFrames++;
  else
    m_sameFrames = 0;
  m_lastFrameHash = h;
  const uint32_t spp = uint32_t(m_samples);
  uint32_t base = 0, count = spp;
  uint32_t accumFlags = 0;
  if (m_sameFrames == 1) {
    // start accumulating: redo the first frame's samples into the buffer
    count = 2 * spp;
    accumFlags = F_ACCUM;
  } else if (m_sameFrames >= 2) {
    base = m_sameFrames * spp;
    accumFlags = F_ACCUM | F_ACCUM_READ;
  }
  fd.size[2] = count;
  fd.size[3] = base;
  fd.flags[0] |= accumFlags;

  out.base = base;
  out.count = count;
  out.accumulate = accumFlags != 0;
  out.depth = wantDepth;
  out.format = fmt;
  out.numPixels = np;
  out.pixelBytes = pixBytes;
  return out;
}

} // namespace gpu
} // namespace tachyon_anari
