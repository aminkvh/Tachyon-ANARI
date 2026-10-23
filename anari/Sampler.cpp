// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"
// std
#include <string>

namespace tachyon_anari {

using helium::math::WrapMode;

Sampler::Sampler(TachyonGlobalState *s) : Object(ANARI_SAMPLER, s) {}

Sampler *Sampler::createInstance(
    std::string_view subtype, TachyonGlobalState *s)
{
  if (subtype == "image1D")
    return new Image1D(s);
  if (subtype == "image2D")
    return new Image2D(s);
  if (subtype == "image3D")
    return new Image3D(s);
  if (subtype == "transform")
    return new TransformSampler(s);
  if (subtype == "primitive")
    return new PrimitiveSampler(s);
  return nullptr; // caller creates an UnsupportedSampler
}

bool Sampler::readInAttribute()
{
  const std::string attr = getParamString("inAttribute", "attribute0");
  if (attr == "color")
    m_inAttribute = 4;
  else if (attr.size() == 10 && attr.compare(0, 9, "attribute") == 0
      && attr[9] >= '0' && attr[9] <= '3')
    m_inAttribute = attr[9] - '0';
  else {
    reportMessage(ANARI_SEVERITY_WARNING,
        "sampler input attribute '%s' not supported",
        attr.c_str());
    return false;
  }
  return true;
}

void Sampler::sampleCallback(const void *user, const float *attr, float *rgb)
{
  const auto *s = static_cast<const Sampler *>(user);
  const float4 v = s->sample(float4(attr[0], attr[1], attr[2], attr[3]));
  rgb[0] = v.x;
  rgb[1] = v.y;
  rgb[2] = v.z;
}

// ImageSampler ///////////////////////////////////////////////////////////////

static WrapMode wrapFromParam(const std::string &s)
{
  return helium::math::wrapModeFromString(s);
}

void ImageSampler::commitCommon(const char *const *wrapNames, int numWrap)
{
  m_valid = readInAttribute();
  m_linear = getParamString("filter", "linear") != "nearest";
  for (int i = 0; i < 3; i++) {
    m_wrap[i] = i < numWrap
        ? wrapFromParam(getParamString(wrapNames[i], "clampToEdge"))
        : WrapMode::DEFAULT;
  }
  m_inTransform = getParam<mat4>("inTransform", mat4(linalg::identity));
  m_inOffset = getParam<float4>("inOffset", float4(0.f));
  m_outTransform = getParam<mat4>("outTransform", mat4(linalg::identity));
  m_outOffset = getParam<float4>("outOffset", float4(0.f));
}

float4 ImageSampler::sample(const float4 &in) const
{
  const float4 coord = linalg::mul(m_inTransform, in) + m_inOffset;
  return linalg::mul(m_outTransform, lookup(coord)) + m_outOffset;
}

static int wrapToInt(WrapMode m)
{
  return m == WrapMode::REPEAT ? 1 : (m == WrapMode::MIRROR_REPEAT ? 2 : 0);
}

static void copyMat(const mat4 &m, float *out)
{
  for (int c = 0; c < 4; c++) {
    out[4 * c + 0] = m[c].x;
    out[4 * c + 1] = m[c].y;
    out[4 * c + 2] = m[c].z;
    out[4 * c + 3] = m[c].w;
  }
}

static void copyVec(const float4 &v, float *out)
{
  out[0] = v.x;
  out[1] = v.y;
  out[2] = v.z;
  out[3] = v.w;
}

void ImageSampler::describeCommon(SamplerDesc &d) const
{
  d.linear = m_linear;
  for (int i = 0; i < 3; i++)
    d.wrap[i] = wrapToInt(m_wrap[i]);
  copyMat(m_inTransform, d.inTransform);
  copyVec(m_inOffset, d.inOffset);
  copyMat(m_outTransform, d.outTransform);
  copyVec(m_outOffset, d.outOffset);
}

static void pushTexel(SamplerDesc &d, const float4 &v)
{
  d.texels.push_back(v.x);
  d.texels.push_back(v.y);
  d.texels.push_back(v.z);
  d.texels.push_back(v.w);
}

bool Image1D::describe(SamplerDesc &d) const
{
  if (!m_valid || !m_image)
    return false;
  describeCommon(d);
  d.dims = 1;
  d.size[0] = int(m_image->size());
  for (int i = 0; i < d.size[0]; i++)
    pushTexel(d, m_image->readAsAttributeValue(i, WrapMode::CLAMP_TO_EDGE));
  return true;
}

bool Image2D::describe(SamplerDesc &d) const
{
  if (!m_valid || !m_image)
    return false;
  describeCommon(d);
  const uint2 sz = m_image->size();
  d.dims = 2;
  d.size[0] = int(sz.x);
  d.size[1] = int(sz.y);
  for (int y = 0; y < d.size[1]; y++)
    for (int x = 0; x < d.size[0]; x++)
      pushTexel(d,
          m_image->readAsAttributeValue(
              {x, y}, WrapMode::CLAMP_TO_EDGE, WrapMode::CLAMP_TO_EDGE));
  return true;
}

bool Image3D::describe(SamplerDesc &d) const
{
  if (!m_valid || !m_image)
    return false;
  describeCommon(d);
  const uint3 sz = m_image->size();
  d.dims = 3;
  d.size[0] = int(sz.x);
  d.size[1] = int(sz.y);
  d.size[2] = int(sz.z);
  for (int z = 0; z < d.size[2]; z++)
    for (int y = 0; y < d.size[1]; y++)
      for (int x = 0; x < d.size[0]; x++)
        pushTexel(d,
            m_image->readAsAttributeValue({x, y, z},
                WrapMode::CLAMP_TO_EDGE,
                WrapMode::CLAMP_TO_EDGE,
                WrapMode::CLAMP_TO_EDGE));
  return true;
}

bool TransformSampler::describe(SamplerDesc &d) const
{
  if (!m_valid)
    return false;
  d.dims = 0;
  copyMat(m_outTransform, d.outTransform);
  copyVec(m_outOffset, d.outOffset);
  return true;
}

// Coordinate along one axis: 'repeat' wraps with fract() before the
// interpolation weights are computed (the neighbor index is wrapped when the
// texel is fetched)
static inline float axisCoord(float v, WrapMode wrap)
{
  return wrap == WrapMode::REPEAT ? helium::math::fract(v) : v;
}

// Image1D ////////////////////////////////////////////////////////////////////

Image1D::Image1D(TachyonGlobalState *s) : ImageSampler(s), m_image(this) {}

void Image1D::commitParameters()
{
  m_image = getParamObject<Array1D>("image");
  static const char *const wrap[] = {"wrapMode"};
  commitCommon(wrap, 1);
}

void Image1D::finalize()
{
  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image1D sampler");
    m_valid = false;
  }
}

float4 Image1D::lookup(const float4 &c) const
{
  const auto i = helium::math::getInterpolant(
      axisCoord(c.x, m_wrap[0]), m_image->size(), true);
  const float4 v0 = m_image->readAsAttributeValue(i.lower, m_wrap[0]);
  const float4 v1 = m_image->readAsAttributeValue(i.upper, m_wrap[0]);
  return m_linear ? linalg::lerp(v0, v1, i.frac) : (i.frac < 0.5f ? v0 : v1);
}

// Image2D ////////////////////////////////////////////////////////////////////

Image2D::Image2D(TachyonGlobalState *s) : ImageSampler(s), m_image(this) {}

void Image2D::commitParameters()
{
  m_image = getParamObject<helium::Array2D>("image");
  static const char *const wrap[] = {"wrapMode1", "wrapMode2"};
  commitCommon(wrap, 2);
}

void Image2D::finalize()
{
  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image2D sampler");
    m_valid = false;
  }
}

float4 Image2D::lookup(const float4 &c) const
{
  const uint2 sz = m_image->size();
  const auto ix = helium::math::getInterpolant(
      axisCoord(c.x, m_wrap[0]), sz.x, true);
  const auto iy = helium::math::getInterpolant(
      axisCoord(c.y, m_wrap[1]), sz.y, true);
  auto at = [&](int32_t x, int32_t y) {
    return m_image->readAsAttributeValue({x, y}, m_wrap[0], m_wrap[1]);
  };
  const float4 v00 = at(ix.lower, iy.lower);
  const float4 v01 = at(ix.lower, iy.upper);
  const float4 v10 = at(ix.upper, iy.lower);
  const float4 v11 = at(ix.upper, iy.upper);
  if (!m_linear) {
    const float4 v0 = iy.frac < 0.5f ? v00 : v01;
    const float4 v1 = iy.frac < 0.5f ? v10 : v11;
    return ix.frac < 0.5f ? v0 : v1;
  }
  return linalg::lerp(
      linalg::lerp(v00, v01, iy.frac), linalg::lerp(v10, v11, iy.frac), ix.frac);
}

// Image3D ////////////////////////////////////////////////////////////////////

Image3D::Image3D(TachyonGlobalState *s) : ImageSampler(s), m_image(this) {}

void Image3D::commitParameters()
{
  m_image = getParamObject<helium::Array3D>("image");
  static const char *const wrap[] = {"wrapMode1", "wrapMode2", "wrapMode3"};
  commitCommon(wrap, 3);
}

void Image3D::finalize()
{
  if (!m_image) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'image' on image3D sampler");
    m_valid = false;
  }
}

float4 Image3D::lookup(const float4 &c) const
{
  const uint3 sz = m_image->size();
  const auto ix = helium::math::getInterpolant(
      axisCoord(c.x, m_wrap[0]), sz.x, true);
  const auto iy = helium::math::getInterpolant(
      axisCoord(c.y, m_wrap[1]), sz.y, true);
  const auto iz = helium::math::getInterpolant(
      axisCoord(c.z, m_wrap[2]), sz.z, true);
  auto at = [&](int32_t x, int32_t y, int32_t z) {
    return m_image->readAsAttributeValue(
        {x, y, z}, m_wrap[0], m_wrap[1], m_wrap[2]);
  };
  const float4 v000 = at(ix.lower, iy.lower, iz.lower);
  const float4 v001 = at(ix.lower, iy.lower, iz.upper);
  const float4 v010 = at(ix.lower, iy.upper, iz.lower);
  const float4 v011 = at(ix.lower, iy.upper, iz.upper);
  const float4 v100 = at(ix.upper, iy.lower, iz.lower);
  const float4 v101 = at(ix.upper, iy.lower, iz.upper);
  const float4 v110 = at(ix.upper, iy.upper, iz.lower);
  const float4 v111 = at(ix.upper, iy.upper, iz.upper);
  if (!m_linear) {
    const float4 v00 = iz.frac < 0.5f ? v000 : v001;
    const float4 v01 = iz.frac < 0.5f ? v010 : v011;
    const float4 v10 = iz.frac < 0.5f ? v100 : v101;
    const float4 v11 = iz.frac < 0.5f ? v110 : v111;
    const float4 v0 = iy.frac < 0.5f ? v00 : v01;
    const float4 v1 = iy.frac < 0.5f ? v10 : v11;
    return ix.frac < 0.5f ? v0 : v1;
  }
  const float4 v00 = linalg::lerp(v000, v001, iz.frac);
  const float4 v01 = linalg::lerp(v010, v011, iz.frac);
  const float4 v10 = linalg::lerp(v100, v101, iz.frac);
  const float4 v11 = linalg::lerp(v110, v111, iz.frac);
  return linalg::lerp(
      linalg::lerp(v00, v01, iy.frac), linalg::lerp(v10, v11, iy.frac), ix.frac);
}

// TransformSampler ///////////////////////////////////////////////////////////

void TransformSampler::commitParameters()
{
  m_valid = readInAttribute();
  // 'transform' is the name older ANARI applications use for 'outTransform'
  const mat4 legacy = getParam<mat4>("transform", mat4(linalg::identity));
  m_outTransform = getParam<mat4>("outTransform", legacy);
  m_outOffset = getParam<float4>("outOffset", float4(0.f));
}

float4 TransformSampler::sample(const float4 &in) const
{
  return linalg::mul(m_outTransform, in) + m_outOffset;
}

// PrimitiveSampler ///////////////////////////////////////////////////////////

PrimitiveSampler::PrimitiveSampler(TachyonGlobalState *s)
    : Sampler(s), m_array(this)
{}

void PrimitiveSampler::commitParameters()
{
  m_array = getParamObject<Array1D>("array");
  // 'offset' is the name older ANARI applications use for 'inOffset'
  const uint64_t legacy = getParam<uint64_t>("offset", 0);
  m_offset = getParam<uint64_t>("inOffset", legacy);
}

void PrimitiveSampler::finalize()
{
  m_valid = m_array.get() != nullptr;
  if (!m_valid) {
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'array' on primitive sampler");
  }
}

float4 PrimitiveSampler::samplePrimitive(size_t prim) const
{
  const uint64_t i = uint64_t(prim) + m_offset;
  if (i >= m_array->size())
    return helium::math::DEFAULT_ATTRIBUTE_VALUE;
  return m_array->readAsAttributeValue(int32_t(i));
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Sampler *);
}
