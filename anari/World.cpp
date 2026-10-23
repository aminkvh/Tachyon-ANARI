// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#include "World.h"
// std
#include <algorithm>
#include <cstring>

namespace tachyon_anari {

// Surface ////////////////////////////////////////////////////////////////////

Surface::Surface(TachyonGlobalState *s) : Object(ANARI_SURFACE, s) {}

void Surface::commitParameters()
{
  m_geometry = getParamObject<Geometry>("geometry");
  m_material = getParamObject<Material>("material");
  m_visible = getParam<bool>("visible", true);
}

void Surface::finalize()
{
  if (!m_material)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'material' on ANARISurface");
  if (!m_geometry)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'geometry' on ANARISurface");
}

bool Surface::isValid() const
{
  return m_geometry && m_geometry->isValid();
}

// Group //////////////////////////////////////////////////////////////////////

Group::Group(TachyonGlobalState *s)
    : Object(ANARI_GROUP, s), m_surfaces(this), m_lights(this)
{}

void Group::commitParameters()
{
  m_surfaces = getParamObject<ObjectArray>("surface");
  m_lights = getParamObject<ObjectArray>("light");
}

// Instance ///////////////////////////////////////////////////////////////////

Instance::Instance(TachyonGlobalState *s)
    : Object(ANARI_INSTANCE, s), m_xfm(linalg::identity)
{}

void Instance::commitParameters()
{
  m_xfm = getParam<mat4>("transform", mat4(linalg::identity));
  m_group = getParamObject<Group>("group");

  static const char *const names[kNumAttributes] = {
      "attribute0", "attribute1", "attribute2", "attribute3", "color"};
  for (int i = 0; i < kNumAttributes; i++) {
    m_attr[i].reset();
    float4 v;
    float3 v3;
    if (getParam(names[i], ANARI_FLOAT32_VEC4, &v))
      m_attr[i] = v;
    else if (getParam(names[i], ANARI_FLOAT32_VEC3, &v3))
      m_attr[i] = float4(v3.x, v3.y, v3.z, 1.f);
  }
}

void Instance::finalize()
{
  if (!m_group)
    reportMessage(ANARI_SEVERITY_WARNING, "missing 'group' on ANARIInstance");
}

// World //////////////////////////////////////////////////////////////////////

World::World(TachyonGlobalState *s)
    : Object(ANARI_WORLD, s),
      m_zeroSurfaces(this),
      m_zeroLights(this),
      m_instances(this)
{}

void World::commitParameters()
{
  m_zeroSurfaces = getParamObject<ObjectArray>("surface");
  m_zeroLights = getParamObject<ObjectArray>("light");
  m_instances = getParamObject<ObjectArray>("instance");
}

static void collectArrays(const ObjectArray *surfaces,
    const ObjectArray *lights,
    const mat4 &xfm,
    const AttributeSet *instAttr,
    std::vector<SurfaceInstance> &outSurfaces,
    std::vector<LightInstance> &outLights)
{
  if (surfaces) {
    std::for_each(surfaces->handlesBegin(), surfaces->handlesEnd(), [&](auto *o) {
      auto *s = static_cast<Surface *>(o);
      if (s && s->isValid() && s->isVisible())
        outSurfaces.push_back({s, xfm, instAttr});
    });
  }
  if (lights) {
    std::for_each(lights->handlesBegin(), lights->handlesEnd(), [&](auto *o) {
      auto *l = static_cast<Light *>(o);
      if (l && l->isValid())
        outLights.push_back({l, xfm});
    });
  }
}

void World::collect(std::vector<SurfaceInstance> &surfaces,
    std::vector<LightInstance> &lights) const
{
  collectArrays(m_zeroSurfaces.get(),
      m_zeroLights.get(),
      mat4(linalg::identity),
      nullptr,
      surfaces,
      lights);

  if (m_instances) {
    std::for_each(
        m_instances->handlesBegin(), m_instances->handlesEnd(), [&](auto *o) {
          auto *i = static_cast<Instance *>(o);
          if (i && i->isValid())
            collectArrays(i->group()->surfaces(),
                i->group()->lights(),
                i->xfm(),
                &i->attributes(),
                surfaces,
                lights);
        });
  }
}

static void mergeBox(
    bool &any, float3 &lo, float3 &hi, const float3 &l, const float3 &h)
{
  lo = any ? linalg::min(lo, l) : l;
  hi = any ? linalg::max(hi, h) : h;
  any = true;
}

// Bounds of the visible, valid surfaces of 'surfaces' under 'xfm'
static bool surfacesBounds(
    const ObjectArray *surfaces, const mat4 &xfm, float3 &lo, float3 &hi)
{
  bool any = false;
  if (!surfaces)
    return false;
  std::for_each(surfaces->handlesBegin(), surfaces->handlesEnd(), [&](auto *o) {
    auto *s = static_cast<Surface *>(o);
    float3 l, h;
    if (s && s->isValid() && s->isVisible()
        && s->geometry()->bounds(xfm, l, h))
      mergeBox(any, lo, hi, l, h);
  });
  return any;
}

bool World::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    // Surfaces directly in the world are bounded exactly. An instance is
    // bounded by the transformed box of its group, which takes O(1) per
    // instance however much geometry the group holds.
    bool any = false;
    float3 lo(0.f), hi(0.f);
    float3 l, h;
    if (surfacesBounds(m_zeroSurfaces.get(), mat4(linalg::identity), l, h))
      mergeBox(any, lo, hi, l, h);
    if (m_instances) {
      std::for_each(
          m_instances->handlesBegin(), m_instances->handlesEnd(), [&](auto *o) {
            auto *i = static_cast<Instance *>(o);
            if (!i || !i->isValid()
                || !surfacesBounds(i->group()->surfaces(),
                    mat4(linalg::identity),
                    l,
                    h))
              return;
            for (int c = 0; c < 8; c++) {
              const float4 p = linalg::mul(i->xfm(),
                  float4((c & 1) ? h.x : l.x,
                      (c & 2) ? h.y : l.y,
                      (c & 4) ? h.z : l.z,
                      1.f));
              const float3 q(p.x, p.y, p.z);
              mergeBox(any, lo, hi, q, q);
            }
          });
    }
    if (!any)
      return false;
    const float3 box[2] = {lo, hi};
    std::memcpy(ptr, box, sizeof(box));
    return true;
  }
  return Object::getProperty(name, type, ptr, size, flags);
}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Surface *);
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Group *);
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Instance *);
ANARI_TYPEFOR_DEFINITION(tachyon_anari::World *);
}
