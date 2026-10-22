// Copyright (c) 2026 Tachyon-dev contributors
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
    std::vector<SurfaceInstance> &outSurfaces,
    std::vector<LightInstance> &outLights)
{
  if (surfaces) {
    std::for_each(surfaces->handlesBegin(), surfaces->handlesEnd(), [&](auto *o) {
      auto *s = static_cast<Surface *>(o);
      if (s && s->isValid() && s->isVisible())
        outSurfaces.push_back({s, xfm});
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
                surfaces,
                lights);
        });
  }
}

bool World::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t size,
    uint32_t flags)
{
  if (name == "bounds" && type == ANARI_FLOAT32_BOX3) {
    std::vector<SurfaceInstance> surfaces;
    std::vector<LightInstance> lights;
    collect(surfaces, lights);
    bool any = false;
    float3 lo(0.f), hi(0.f);
    for (auto &si : surfaces) {
      float3 l, h;
      if (si.surface->geometry()->bounds(si.xfm, l, h)) {
        lo = any ? linalg::min(lo, l) : l;
        hi = any ? linalg::max(hi, h) : h;
        any = true;
      }
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
