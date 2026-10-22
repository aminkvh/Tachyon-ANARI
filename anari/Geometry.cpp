// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "Geometry.h"
// std
#include <algorithm>
#include <cmath>
#include <string>

namespace tachyon_anari {

Geometry::Geometry(TachyonGlobalState *s) : Object(ANARI_GEOMETRY, s) {}

Geometry *Geometry::createInstance(
    std::string_view subtype, TachyonGlobalState *s)
{
  if (subtype == "triangle")
    return new Triangle(s);
  return nullptr; // caller creates an UnsupportedGeometry
}

static const char *const kAttrNames[kNumAttributes] = {
    "attribute0", "attribute1", "attribute2", "attribute3", "color"};

void Geometry::commitParameters()
{
  for (int i = 0; i < kNumAttributes; i++) {
    m_uniform[i].reset();
    float4 v;
    float3 v3;
    if (getParam(kAttrNames[i], ANARI_FLOAT32_VEC4, &v))
      m_uniform[i] = v;
    else if (getParam(kAttrNames[i], ANARI_FLOAT32_VEC3, &v3))
      m_uniform[i] = float4(v3.x, v3.y, v3.z, 1.f);
  }
}

// Triangle ///////////////////////////////////////////////////////////////////

Triangle::Triangle(TachyonGlobalState *s)
    : Geometry(s),
      m_index(this),
      m_position(this),
      m_normal(this),
      m_vertexAttr{{{this}, {this}, {this}, {this}, {this}}},
      m_primAttr{{{this}, {this}, {this}, {this}, {this}}}
{}

void Triangle::commitParameters()
{
  Geometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_position = getParamObject<Array1D>("vertex.position");
  m_normal = getParamObject<Array1D>("vertex.normal");
  for (int i = 0; i < kNumAttributes; i++) {
    m_vertexAttr[i] = getParamObject<Array1D>(
        (std::string("vertex.") + kAttrNames[i]).c_str());
    m_primAttr[i] = getParamObject<Array1D>(
        (std::string("primitive.") + kAttrNames[i]).c_str());
  }
}

void Triangle::finalize()
{
  m_valid = false;
  m_numTriangles = 0;

  if (!m_position) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on triangle geometry");
    return;
  }
  if (m_position->elementType() != ANARI_FLOAT32_VEC3) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'vertex.position' must be FLOAT32_VEC3 (got %s)",
        anari::toString(m_position->elementType()));
    return;
  }
  if (m_index && m_index->elementType() != ANARI_UINT32_VEC3) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'primitive.index' must be UINT32_VEC3 (got %s), ignoring it",
        anari::toString(m_index->elementType()));
    m_index = nullptr;
  }
  if (m_normal && m_normal->elementType() != ANARI_FLOAT32_VEC3) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'vertex.normal' must be FLOAT32_VEC3 (got %s), ignoring it",
        anari::toString(m_normal->elementType()));
    m_normal = nullptr;
  }
  if (m_normal && m_normal->size() < m_position->size()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'vertex.normal' has fewer elements than 'vertex.position', "
        "ignoring it");
    m_normal = nullptr;
  }
  m_numTriangles = m_index ? m_index->size() : m_position->size() / 3;
  for (int i = 0; i < kNumAttributes; i++) {
    if (m_vertexAttr[i] && m_vertexAttr[i]->size() < m_position->size()) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "'vertex.%s' has fewer elements than 'vertex.position', ignoring it",
          kAttrNames[i]);
      m_vertexAttr[i] = nullptr;
    }
    if (m_primAttr[i] && m_primAttr[i]->size() < m_numTriangles)
      m_primAttr[i] = nullptr;
  }
  m_valid = true;
}

ColorSource Triangle::attributeSource(int attr) const
{
  if (m_vertexAttr[attr])
    return ColorSource::VERTEX;
  if (m_primAttr[attr])
    return ColorSource::PRIMITIVE;
  return Geometry::attributeSource(attr);
}

static inline void xfmPoint(const mat4 &m, const float *p, float *out)
{
  const float4 r = linalg::mul(m, float4(p[0], p[1], p[2], 1.f));
  out[0] = r.x;
  out[1] = r.y;
  out[2] = r.z;
}

static inline void xfmNormal(const float3x3 &nm, const float *n, float *out)
{
  float3 r = linalg::mul(nm, float3(n[0], n[1], n[2]));
  const float len = linalg::length(r);
  if (len > 0.f)
    r /= len;
  out[0] = r.x;
  out[1] = r.y;
  out[2] = r.z;
}

// Reads a color element as rgb regardless of its stored type
static inline void readColor(const Array1D &a, size_t i, float *out)
{
  const float4 c = a.readAsAttributeValue(int32_t(i));
  out[0] = c.x;
  out[1] = c.y;
  out[2] = c.z;
}

void Triangle::emit(
    TachyonScene &scene, int style, const mat4 &xfm, int colorAttr) const
{
  if (!m_valid)
    return;

  const float *pos = static_cast<const float *>(m_position->begin());
  const size_t numVerts = m_position->size();
  const uint32_t *idx =
      m_index ? static_cast<const uint32_t *>(m_index->begin()) : nullptr;
  const float *nrm =
      m_normal ? static_cast<const float *>(m_normal->begin()) : nullptr;

  // Normals transform with the inverse transpose of the upper 3x3
  float3x3 normalMat = float3x3(linalg::identity);
  if (nrm) {
    const float3x3 m3(float3(xfm[0].x, xfm[0].y, xfm[0].z),
        float3(xfm[1].x, xfm[1].y, xfm[1].z),
        float3(xfm[2].x, xfm[2].y, xfm[2].z));
    normalMat = linalg::transpose(linalg::inverse(m3));
  }

  const Array1D *vattr =
      colorAttr >= 0 ? m_vertexAttr[colorAttr].get() : nullptr;
  const Array1D *pattr =
      (colorAttr >= 0 && !vattr) ? m_primAttr[colorAttr].get() : nullptr;
  const bool vcol = vattr != nullptr;
  const bool pcol = pattr != nullptr;

  for (size_t t = 0; t < m_numTriangles; t++) {
    const uint32_t i0 = idx ? idx[3 * t + 0] : uint32_t(3 * t + 0);
    const uint32_t i1 = idx ? idx[3 * t + 1] : uint32_t(3 * t + 1);
    const uint32_t i2 = idx ? idx[3 * t + 2] : uint32_t(3 * t + 2);
    if (i0 >= numVerts || i1 >= numVerts || i2 >= numVerts)
      continue;

    float p[9], n[9], c[9];
    xfmPoint(xfm, pos + 3 * i0, p + 0);
    xfmPoint(xfm, pos + 3 * i1, p + 3);
    xfmPoint(xfm, pos + 3 * i2, p + 6);

    if (nrm) {
      xfmNormal(normalMat, nrm + 3 * i0, n + 0);
      xfmNormal(normalMat, nrm + 3 * i1, n + 3);
      xfmNormal(normalMat, nrm + 3 * i2, n + 6);
    }

    if (vcol) {
      readColor(*vattr, i0, c + 0);
      readColor(*vattr, i1, c + 3);
      readColor(*vattr, i2, c + 6);
    } else if (pcol) {
      readColor(*pattr, t, c + 0);
      std::copy(c, c + 3, c + 3);
      std::copy(c, c + 3, c + 6);
    }

    scene.addTriangle(
        style, p, nrm ? n : nullptr, (vcol || pcol) ? c : nullptr);
  }
}

bool Triangle::bounds(const mat4 &xfm, float3 &lo, float3 &hi) const
{
  if (!m_valid || m_position->size() == 0)
    return false;
  const float *pos = static_cast<const float *>(m_position->begin());
  const size_t n = m_position->size();
  for (size_t i = 0; i < n; i++) {
    float p[3];
    xfmPoint(xfm, pos + 3 * i, p);
    const float3 v(p[0], p[1], p[2]);
    lo = (i == 0) ? v : linalg::min(lo, v);
    hi = (i == 0) ? v : linalg::max(hi, v);
  }
  return true;
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Geometry *);
}
