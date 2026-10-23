// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "SceneBackend.h"
#include "helium/array/Array2D.h"
#include "helium/array/Array3D.h"

namespace tachyon_anari {

constexpr int kNumSamplerAttributes = 5; // attribute0-3 and color

// A sampler maps the value of a surface attribute (or the primitive index) to
// a color. Geometries feed it the attribute values; the result of
// sample()/samplePrimitive() is a (r, g, b, a) value.
struct Sampler : public Object
{
  Sampler(TachyonGlobalState *s);

  static Sampler *createInstance(
      std::string_view subtype, TachyonGlobalState *s);

  bool affectsScene() const override { return true; }
  bool isValid() const override { return m_valid; }

  // Attribute (0-3, 4 = color) the sampler reads
  int inAttribute() const { return m_inAttribute; }
  // Primitive samplers read the primitive index instead of an attribute
  virtual bool isPrimitive() const { return false; }

  // Sampled value for the (interpolated) input attribute value 'in'
  virtual float4 sample(const float4 &in) const { return in; }
  // Sampled value of primitive 'prim' (primitive samplers only)
  virtual float4 samplePrimitive(size_t) const
  {
    return helium::math::DEFAULT_ATTRIBUTE_VALUE;
  }

  // Fills 'd' with what a shader needs to evaluate this sampler on an input
  // attribute value; false when that is not possible (primitive samplers are
  // evaluated per primitive on the CPU).
  virtual bool describe(SamplerDesc &) const { return false; }

  // Callback for Tachyon's per-hit texture function (TachyonScene's
  // AttrSampleFn): 'user' is the Sampler, 'attr' the 4 component input value.
  static void sampleCallback(const void *user, const float *attr, float *rgb);

 protected:
  // Reads the string parameter 'inAttribute' (default attribute0) into
  // m_inAttribute; returns false for attributes this device cannot provide.
  bool readInAttribute();

  int m_inAttribute{0};
  bool m_valid{false};
};

// Common part of the image samplers: filtering, wrapping and the transforms
// applied before (inTransform, inOffset) and after (outTransform, outOffset)
// the image lookup.
struct ImageSampler : public Sampler
{
  ImageSampler(TachyonGlobalState *s) : Sampler(s) {}

  float4 sample(const float4 &in) const override;

 protected:
  void describeCommon(SamplerDesc &d) const;
  void commitCommon(const char *const *wrapNames, int numWrap);
  // Lookup at the (transformed) texture coordinate
  virtual float4 lookup(const float4 &coord) const = 0;

  helium::math::WrapMode m_wrap[3]{helium::math::WrapMode::DEFAULT,
      helium::math::WrapMode::DEFAULT,
      helium::math::WrapMode::DEFAULT};
  bool m_linear{true};
  mat4 m_inTransform{linalg::identity};
  float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
  mat4 m_outTransform{linalg::identity};
  float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

struct Image1D : public ImageSampler
{
  Image1D(TachyonGlobalState *s);
  void commitParameters() override;
  void finalize() override;
  bool describe(SamplerDesc &d) const override;

 private:
  float4 lookup(const float4 &coord) const override;
  helium::ChangeObserverPtr<Array1D> m_image;
};

struct Image2D : public ImageSampler
{
  Image2D(TachyonGlobalState *s);
  void commitParameters() override;
  void finalize() override;
  bool describe(SamplerDesc &d) const override;

 private:
  float4 lookup(const float4 &coord) const override;
  helium::ChangeObserverPtr<helium::Array2D> m_image;
};

struct Image3D : public ImageSampler
{
  Image3D(TachyonGlobalState *s);
  void commitParameters() override;
  void finalize() override;
  bool describe(SamplerDesc &d) const override;

 private:
  float4 lookup(const float4 &coord) const override;
  helium::ChangeObserverPtr<helium::Array3D> m_image;
};

// outTransform * attribute + outOffset
struct TransformSampler : public Sampler
{
  TransformSampler(TachyonGlobalState *s) : Sampler(s) {}
  void commitParameters() override;
  float4 sample(const float4 &in) const override;
  bool describe(SamplerDesc &d) const override;

 private:
  mat4 m_outTransform{linalg::identity};
  float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
};

// array[primitiveIndex + inOffset]
struct PrimitiveSampler : public Sampler
{
  PrimitiveSampler(TachyonGlobalState *s);
  void commitParameters() override;
  void finalize() override;

  bool isPrimitive() const override { return true; }
  float4 samplePrimitive(size_t prim) const override;

 private:
  helium::ChangeObserverPtr<Array1D> m_array;
  uint64_t m_offset{0};
};

using UnsupportedSampler = Unsupported<Sampler>;

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Sampler *, ANARI_SAMPLER);
}
