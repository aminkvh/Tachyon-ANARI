// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#include "Geometry.h"
// std
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace tachyon_anari {

Geometry::Geometry(TachyonGlobalState *s) : Object(ANARI_GEOMETRY, s) {}

Geometry *Geometry::createInstance(
    std::string_view subtype, TachyonGlobalState *s)
{
  if (subtype == "triangle")
    return new Triangle(s);
  if (subtype == "sphere")
    return new Sphere(s);
  if (subtype == "cylinder")
    return new Cylinder(s);
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

// Reads a color element as rgb regardless of its stored type; with a sampler
// the element is the sampler's input attribute value
static inline void readColor(
    const Array1D &a, size_t i, const Sampler *sampler, float *out)
{
  float4 c = a.readAsAttributeValue(int32_t(i));
  if (sampler)
    c = sampler->sample(c);
  out[0] = c.x;
  out[1] = c.y;
  out[2] = c.z;
}

// Color of primitive 'prim' from a primitive sampler
static inline void primitiveColor(
    const Sampler &sampler, size_t prim, float *out)
{
  const float4 c = sampler.samplePrimitive(prim);
  out[0] = c.x;
  out[1] = c.y;
  out[2] = c.z;
}

using AttrArrays = std::array<helium::ChangeObserverPtr<Array1D>, kNumAttributes>;

// Values of the attributes in 'mask' for primitive 'prim' made of vertices
// verts[0..numVerts): a vertex attribute is averaged over the vertices, a
// primitive attribute is read directly.
static void fillAttrValues(unsigned mask,
    const AttrArrays &vertexAttr,
    const AttrArrays &primAttr,
    const size_t *verts,
    int numVerts,
    size_t prim,
    float4 *vals)
{
  for (int a = 0; a < kNumAttributes; a++) {
    if (!(mask & (1u << a)))
      continue;
    if (vertexAttr[a]) {
      float4 sum(0.f);
      for (int v = 0; v < numVerts; v++)
        sum += vertexAttr[a]->readAsAttributeValue(int32_t(verts[v]));
      vals[a] = sum / float(numVerts);
    } else if (primAttr[a]) {
      vals[a] = primAttr[a]->readAsAttributeValue(int32_t(prim));
    } else {
      vals[a] = float4(0.f, 0.f, 0.f, 1.f);
    }
  }
}

void Triangle::emit(SceneBackend &scene,
    int style,
    const mat4 &xfm,
    int colorAttr,
    PrimStyleBinder *binder,
    const Sampler *colorSampler) const
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
  const bool primSampler = colorSampler && colorSampler->isPrimitive();
  // A sampler applied to a vertex attribute is evaluated per hit, with the
  // attribute interpolated over the triangle
  const bool hitSampled = colorSampler && !primSampler && vattr;
  const Sampler *valueSampler = (colorSampler && !primSampler && !vattr)
      ? colorSampler
      : nullptr; // maps constant per-primitive values
  // what a GPU backend needs to evaluate the sampler per hit
  SamplerDesc samplerDesc;
  const bool haveDesc = hitSampled && colorSampler->describe(samplerDesc);
  const bool vcol = vattr != nullptr && !hitSampled;
  const bool pcol = pattr != nullptr || primSampler;

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
      readColor(*vattr, i0, nullptr, c + 0);
      readColor(*vattr, i1, nullptr, c + 3);
      readColor(*vattr, i2, nullptr, c + 6);
    } else if (pcol) {
      if (primSampler)
        primitiveColor(*colorSampler, t, c + 0);
      else
        readColor(*pattr, t, valueSampler, c + 0);
      std::copy(c, c + 3, c + 3);
      std::copy(c, c + 3, c + 6);
    }

    int tstyle = style;
    if (binder) {
      const size_t verts[3] = {i0, i1, i2};
      float4 vals[kNumAttributes];
      fillAttrValues(binder->attrMask, m_vertexAttr, m_primAttr, verts, 3, t, vals);
      tstyle = binder->styleFor(vals, t);
      if (tstyle < 0)
        continue;
    }
    if (hitSampled) {
      float a[12];
      const uint32_t vi[3] = {i0, i1, i2};
      for (int k = 0; k < 3; k++) {
        const float4 v = vattr->readAsAttributeValue(int32_t(vi[k]));
        a[4 * k + 0] = v.x;
        a[4 * k + 1] = v.y;
        a[4 * k + 2] = v.z;
        a[4 * k + 3] = v.w;
      }
      scene.addSampledTriangle(tstyle,
          p,
          nrm ? n : nullptr,
          a,
          Sampler::sampleCallback,
          colorSampler,
          haveDesc ? &samplerDesc : nullptr);
      continue;
    }
    scene.addTriangle(
        tstyle, p, nrm ? n : nullptr, (vcol || pcol) ? c : nullptr);
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

// AttributedGeometry /////////////////////////////////////////////////////////

AttributedGeometry::AttributedGeometry(TachyonGlobalState *s)
    : Geometry(s),
      m_position(this),
      m_vertexAttr{{{this}, {this}, {this}, {this}, {this}}},
      m_primAttr{{{this}, {this}, {this}, {this}, {this}}}
{}

void AttributedGeometry::commitParameters()
{
  Geometry::commitParameters();
  m_position = getParamObject<Array1D>("vertex.position");
  for (int i = 0; i < kNumAttributes; i++) {
    m_vertexAttr[i] = getParamObject<Array1D>(
        (std::string("vertex.") + kAttrNames[i]).c_str());
    m_primAttr[i] = getParamObject<Array1D>(
        (std::string("primitive.") + kAttrNames[i]).c_str());
  }
}

bool AttributedGeometry::validatePositions(const char *what)
{
  m_valid = false;
  m_numPrims = 0;
  if (!m_position) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'vertex.position' on %s geometry",
        what);
    return false;
  }
  if (m_position->elementType() != ANARI_FLOAT32_VEC3) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'vertex.position' must be FLOAT32_VEC3 (got %s)",
        anari::toString(m_position->elementType()));
    return false;
  }
  return true;
}

void AttributedGeometry::validateAttributes(size_t numPrims)
{
  m_numPrims = numPrims;
  for (int i = 0; i < kNumAttributes; i++) {
    if (m_vertexAttr[i] && m_vertexAttr[i]->size() < m_position->size()) {
      reportMessage(ANARI_SEVERITY_WARNING,
          "'vertex.%s' has fewer elements than 'vertex.position', ignoring it",
          kAttrNames[i]);
      m_vertexAttr[i] = nullptr;
    }
    if (m_primAttr[i] && m_primAttr[i]->size() < m_numPrims)
      m_primAttr[i] = nullptr;
  }
}

ColorSource AttributedGeometry::attributeSource(int attr) const
{
  if (m_vertexAttr[attr])
    return ColorSource::VERTEX;
  if (m_primAttr[attr])
    return ColorSource::PRIMITIVE;
  return Geometry::attributeSource(attr);
}

size_t AttributedGeometry::indexAt(
    const Array1D &index, size_t i, int comps, int c)
{
  const ANARIDataType t = index.elementType();
  if (t == ANARI_UINT64 || t == ANARI_UINT64_VEC2)
    return size_t(static_cast<const uint64_t *>(index.begin())[comps * i + c]);
  return size_t(static_cast<const uint32_t *>(index.begin())[comps * i + c]);
}

// Scale applied to radii by a transform (exact for uniform scale)
static float radiusScale(const mat4 &m)
{
  const float3x3 m3(float3(m[0].x, m[0].y, m[0].z),
      float3(m[1].x, m[1].y, m[1].z),
      float3(m[2].x, m[2].y, m[2].z));
  const float d = std::abs(linalg::determinant(m3));
  return d > 0.f ? std::cbrt(d) : 0.f;
}

// Sphere /////////////////////////////////////////////////////////////////////

Sphere::Sphere(TachyonGlobalState *s)
    : AttributedGeometry(s), m_index(this), m_radius(this)
{}

void Sphere::commitParameters()
{
  AttributedGeometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("vertex.radius");
  m_globalRadius = getParam<float>("radius", 1.f);
}

void Sphere::finalize()
{
  if (!validatePositions("sphere"))
    return;
  if (m_index && m_index->elementType() != ANARI_UINT32
      && m_index->elementType() != ANARI_UINT64) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'primitive.index' must be UINT32 or UINT64 (got %s), ignoring it",
        anari::toString(m_index->elementType()));
    m_index = nullptr;
  }
  if (m_radius && m_radius->elementType() != ANARI_FLOAT32) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'vertex.radius' must be FLOAT32 (got %s), ignoring it",
        anari::toString(m_radius->elementType()));
    m_radius = nullptr;
  }
  if (m_radius && m_radius->size() < m_position->size()) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'vertex.radius' has fewer elements than 'vertex.position', "
        "ignoring it");
    m_radius = nullptr;
  }
  validateAttributes(m_index ? m_index->size() : m_position->size());
  m_valid = true;
}

void Sphere::emit(SceneBackend &scene,
    int style,
    const mat4 &xfm,
    int colorAttr,
    PrimStyleBinder *binder,
    const Sampler *colorSampler) const
{
  if (!m_valid)
    return;

  const float *pos = static_cast<const float *>(m_position->begin());
  const size_t numVerts = m_position->size();
  const float *vrad =
      m_radius ? static_cast<const float *>(m_radius->begin()) : nullptr;
  const float rs = radiusScale(xfm);

  const Array1D *vattr =
      colorAttr >= 0 ? m_vertexAttr[colorAttr].get() : nullptr;
  const Array1D *pattr =
      (colorAttr >= 0 && !vattr) ? m_primAttr[colorAttr].get() : nullptr;
  const bool primSampler = colorSampler && colorSampler->isPrimitive();
  const Sampler *valueSampler = primSampler ? nullptr : colorSampler;
  const bool perSphereColor = vattr || pattr || primSampler;
  auto sphereColor = [&](size_t vi, size_t p, float *col) {
    if (primSampler)
      primitiveColor(*colorSampler, p, col);
    else
      readColor(vattr ? *vattr : *pattr, vattr ? vi : p, valueSampler, col);
  };

  // Fast path: positions used in place (no index, identity transform) and
  // radius/validity needing no per-sphere work
  const bool identity = xfm == mat4(linalg::identity);
  const bool direct = !m_index && identity && m_numPrims <= numVerts;
  const bool needRadii = vrad != nullptr && rs != 1.f;

  std::vector<float> tmpPos, tmpRad, tmpCol;
  const float *outPos = pos;
  const float *outRad = vrad;
  bool compact = !direct || binder; // need the gathered/transformed arrays
  std::vector<int> tmpStyle; // per kept sphere, with a binder

  if (vrad && direct && !needRadii) {
    for (size_t p = 0; p < m_numPrims && !compact; p++)
      if (!(vrad[p] > 0.f))
        compact = true;
  }
  if (!vrad && !(m_globalRadius * rs > 0.f))
    return;
  if (needRadii)
    compact = true;

  if (compact) {
    tmpPos.reserve(3 * m_numPrims);
    if (vrad)
      tmpRad.reserve(m_numPrims);
    if (perSphereColor)
      tmpCol.reserve(3 * m_numPrims);
    for (size_t p = 0; p < m_numPrims; p++) {
      const size_t vi = m_index ? indexAt(*m_index, p, 1, 0) : p;
      if (vi >= numVerts)
        continue;
      float r = 0.f;
      if (vrad) {
        r = vrad[vi] * rs;
        if (!(r > 0.f))
          continue;
      }
      if (binder) {
        float4 vals[kNumAttributes];
        fillAttrValues(binder->attrMask, m_vertexAttr, m_primAttr, &vi, 1, p, vals);
        const int sid = binder->styleFor(vals, p);
        if (sid < 0)
          continue;
        tmpStyle.push_back(sid);
      }
      float c[3];
      xfmPoint(xfm, pos + 3 * vi, c);
      tmpPos.insert(tmpPos.end(), c, c + 3);
      if (vrad)
        tmpRad.push_back(r);
      if (perSphereColor) {
        float col[3];
        sphereColor(vi, p, col);
        tmpCol.insert(tmpCol.end(), col, col + 3);
      }
    }
    outPos = tmpPos.data();
    outRad = vrad ? tmpRad.data() : nullptr;
    if (binder) {
      // one sphere array per distinct style
      std::map<int, std::vector<size_t>> groups;
      for (size_t i = 0; i < tmpStyle.size(); i++)
        groups[tmpStyle[i]].push_back(i);
      std::vector<float> gPos, gRad, gCol;
      for (auto &g : groups) {
        const size_t n = g.second.size();
        gPos.resize(3 * n);
        gRad.resize(vrad ? n : 0);
        gCol.resize(tmpCol.empty() ? 0 : 3 * n);
        for (size_t k = 0; k < n; k++) {
          const size_t i = g.second[k];
          std::copy(&tmpPos[3 * i], &tmpPos[3 * i] + 3, &gPos[3 * k]);
          if (vrad)
            gRad[k] = tmpRad[i];
          if (!tmpCol.empty())
            std::copy(&tmpCol[3 * i], &tmpCol[3 * i] + 3, &gCol[3 * k]);
        }
        scene.addSpheres(g.first,
            n,
            gPos.data(),
            vrad ? gRad.data() : nullptr,
            m_globalRadius * rs,
            gCol.empty() ? nullptr : gCol.data());
      }
      return;
    }
    scene.addSpheres(style,
        tmpPos.size() / 3,
        outPos,
        outRad,
        m_globalRadius * rs,
        tmpCol.empty() ? nullptr : tmpCol.data());
    return;
  }

  // direct: only per-sphere colors still need converting
  if (perSphereColor) {
    tmpCol.resize(3 * m_numPrims);
    for (size_t p = 0; p < m_numPrims; p++)
      sphereColor(p, p, &tmpCol[3 * p]);
  }
  scene.addSpheres(style,
      m_numPrims,
      outPos,
      outRad,
      m_globalRadius * rs,
      tmpCol.empty() ? nullptr : tmpCol.data());
}

bool Sphere::bounds(const mat4 &xfm, float3 &lo, float3 &hi) const
{
  if (!m_valid)
    return false;
  const float *pos = static_cast<const float *>(m_position->begin());
  const size_t numVerts = m_position->size();
  const float *vrad =
      m_radius ? static_cast<const float *>(m_radius->begin()) : nullptr;
  const float rs = radiusScale(xfm);
  bool any = false;
  for (size_t p = 0; p < m_numPrims; p++) {
    const size_t vi = m_index ? indexAt(*m_index, p, 1, 0) : p;
    if (vi >= numVerts)
      continue;
    const float r = (vrad ? vrad[vi] : m_globalRadius) * rs;
    if (!(r > 0.f))
      continue;
    float c[3];
    xfmPoint(xfm, pos + 3 * vi, c);
    const float3 v(c[0], c[1], c[2]);
    lo = any ? linalg::min(lo, v - r) : v - r;
    hi = any ? linalg::max(hi, v + r) : v + r;
    any = true;
  }
  return any;
}

// Cylinder ///////////////////////////////////////////////////////////////////

Cylinder::Cylinder(TachyonGlobalState *s)
    : AttributedGeometry(s), m_index(this), m_radius(this), m_vertexCap(this)
{}

void Cylinder::commitParameters()
{
  AttributedGeometry::commitParameters();
  m_index = getParamObject<Array1D>("primitive.index");
  m_radius = getParamObject<Array1D>("primitive.radius");
  m_vertexCap = getParamObject<Array1D>("vertex.cap");
  m_globalRadius = getParam<float>("radius", 1.f);

  const std::string caps = getParamString("caps", "none");
  m_globalCaps = 0;
  if (caps == "first")
    m_globalCaps = 1;
  else if (caps == "second")
    m_globalCaps = 2;
  else if (caps == "both")
    m_globalCaps = 3;
  else if (caps != "none") {
    reportMessage(ANARI_SEVERITY_WARNING,
        "unknown cylinder 'caps' value '%s', using 'none'",
        caps.c_str());
  }
}

void Cylinder::finalize()
{
  if (!validatePositions("cylinder"))
    return;
  if (m_index && m_index->elementType() != ANARI_UINT32_VEC2
      && m_index->elementType() != ANARI_UINT64_VEC2) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'primitive.index' must be UINT32_VEC2 or UINT64_VEC2 (got %s), "
        "ignoring it",
        anari::toString(m_index->elementType()));
    m_index = nullptr;
  }
  if (m_radius && m_radius->elementType() != ANARI_FLOAT32) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'primitive.radius' must be FLOAT32 (got %s), ignoring it",
        anari::toString(m_radius->elementType()));
    m_radius = nullptr;
  }
  if (m_vertexCap
      && (m_vertexCap->elementType() != ANARI_UINT8
          || m_vertexCap->size() < m_position->size())) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "'vertex.cap' must be UINT8 with one element per vertex, ignoring it");
    m_vertexCap = nullptr;
  }
  validateAttributes(m_index ? m_index->size() : m_position->size() / 2);
  if (m_radius && m_radius->size() < m_numPrims)
    m_radius = nullptr;
  m_valid = true;
}

void Cylinder::emit(SceneBackend &scene,
    int style,
    const mat4 &xfm,
    int colorAttr,
    PrimStyleBinder *binder,
    const Sampler *colorSampler) const
{
  if (!m_valid)
    return;

  const float *pos = static_cast<const float *>(m_position->begin());
  const size_t numVerts = m_position->size();
  const float *prad =
      m_radius ? static_cast<const float *>(m_radius->begin()) : nullptr;
  const uint8_t *vcap = m_vertexCap
      ? static_cast<const uint8_t *>(m_vertexCap->begin())
      : nullptr;
  const float rs = radiusScale(xfm);

  const Array1D *vattr =
      colorAttr >= 0 ? m_vertexAttr[colorAttr].get() : nullptr;
  const Array1D *pattr =
      (colorAttr >= 0 && !vattr) ? m_primAttr[colorAttr].get() : nullptr;
  const bool primSampler = colorSampler && colorSampler->isPrimitive();
  const Sampler *valueSampler = primSampler ? nullptr : colorSampler;

  for (size_t p = 0; p < m_numPrims; p++) {
    const size_t v0 = m_index ? indexAt(*m_index, p, 2, 0) : 2 * p;
    const size_t v1 = m_index ? indexAt(*m_index, p, 2, 1) : 2 * p + 1;
    if (v0 >= numVerts || v1 >= numVerts)
      continue;
    const float r = (prad ? prad[p] : m_globalRadius) * rs;
    if (!(r > 0.f))
      continue;

    int cstyle = style;
    if (binder) {
      const size_t verts[2] = {v0, v1};
      float4 vals[kNumAttributes];
      fillAttrValues(binder->attrMask, m_vertexAttr, m_primAttr, verts, 2, p, vals);
      cstyle = binder->styleFor(vals, p);
      if (cstyle < 0)
        continue;
    }

    float a[3], b[3], c0[3], c1[3];
    xfmPoint(xfm, pos + 3 * v0, a);
    xfmPoint(xfm, pos + 3 * v1, b);
    const float *col0 = nullptr, *col1 = nullptr;
    if (vattr) {
      // the colors at both ends; the cylinder is split between them
      readColor(*vattr, v0, valueSampler, c0);
      readColor(*vattr, v1, valueSampler, c1);
      col0 = c0;
      col1 = c1;
    } else if (pattr) {
      readColor(*pattr, p, valueSampler, c0);
      col0 = c0;
    } else if (primSampler) {
      primitiveColor(*colorSampler, p, c0);
      col0 = c0;
    }

    int caps = m_globalCaps;
    if (vcap) {
      caps |= vcap[v0] ? 1 : 0;
      caps |= vcap[v1] ? 2 : 0;
    }
    scene.addCylinder(cstyle, a, b, r, col0, col1, caps);
  }
}

bool Cylinder::bounds(const mat4 &xfm, float3 &lo, float3 &hi) const
{
  if (!m_valid)
    return false;
  const float *pos = static_cast<const float *>(m_position->begin());
  const size_t numVerts = m_position->size();
  const float *prad =
      m_radius ? static_cast<const float *>(m_radius->begin()) : nullptr;
  const float rs = radiusScale(xfm);
  bool any = false;
  for (size_t p = 0; p < m_numPrims; p++) {
    const size_t v0 = m_index ? indexAt(*m_index, p, 2, 0) : 2 * p;
    const size_t v1 = m_index ? indexAt(*m_index, p, 2, 1) : 2 * p + 1;
    if (v0 >= numVerts || v1 >= numVerts)
      continue;
    const float r = (prad ? prad[p] : m_globalRadius) * rs;
    if (!(r > 0.f))
      continue;
    float a[3], b[3];
    xfmPoint(xfm, pos + 3 * v0, a);
    xfmPoint(xfm, pos + 3 * v1, b);
    const float3 va(a[0], a[1], a[2]), vb(b[0], b[1], b[2]);
    // conservative: the end disks extend at most r along every axis
    const float3 mn = linalg::min(va, vb) - r;
    const float3 mx = linalg::max(va, vb) + r;
    lo = any ? linalg::min(lo, mn) : mn;
    hi = any ? linalg::max(hi, mx) : mx;
    any = true;
  }
  return any;
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Geometry *);
}
