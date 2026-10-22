// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Geometry.h"
#include "Light.h"
#include "Material.h"
#include "Object.h"
// std
#include <vector>

namespace tachyon_anari {

struct Surface : public Object
{
  Surface(TachyonGlobalState *s);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override;
  bool affectsScene() const override { return true; }

  bool isVisible() const { return m_visible; }
  const Geometry *geometry() const { return m_geometry.ptr; }
  const Material *material() const { return m_material.ptr; }

 private:
  bool m_visible{true};
  helium::IntrusivePtr<Geometry> m_geometry;
  helium::IntrusivePtr<Material> m_material;
};

struct Group : public Object
{
  Group(TachyonGlobalState *s);

  void commitParameters() override;
  bool affectsScene() const override { return true; }

  const ObjectArray *surfaces() const { return m_surfaces.get(); }
  const ObjectArray *lights() const { return m_lights.get(); }

 private:
  helium::ChangeObserverPtr<ObjectArray> m_surfaces;
  helium::ChangeObserverPtr<ObjectArray> m_lights;
};

struct Instance : public Object
{
  Instance(TachyonGlobalState *s);

  void commitParameters() override;
  void finalize() override;
  bool isValid() const override { return m_group; }
  bool affectsScene() const override { return true; }

  const Group *group() const { return m_group.ptr; }
  const mat4 &xfm() const { return m_xfm; }

 private:
  mat4 m_xfm;
  helium::IntrusivePtr<Group> m_group;
};

struct SurfaceInstance
{
  const Surface *surface;
  mat4 xfm;
};

struct LightInstance
{
  const Light *light;
  mat4 xfm;
};

struct World : public Object
{
  World(TachyonGlobalState *s);

  bool getProperty(const std::string_view &name,
      ANARIDataType type,
      void *ptr,
      uint64_t size,
      uint32_t flags) override;

  void commitParameters() override;
  bool affectsScene() const override { return true; }

  // Flattens the scene graph into valid surfaces and lights with the full
  // instance transform applied.
  void collect(std::vector<SurfaceInstance> &surfaces,
      std::vector<LightInstance> &lights) const;

 private:
  helium::ChangeObserverPtr<ObjectArray> m_zeroSurfaces;
  helium::ChangeObserverPtr<ObjectArray> m_zeroLights;
  helium::ChangeObserverPtr<ObjectArray> m_instances;
};

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Surface *, ANARI_SURFACE);
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Group *, ANARI_GROUP);
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::Instance *, ANARI_INSTANCE);
ANARI_TYPEFOR_SPECIALIZATION(tachyon_anari::World *, ANARI_WORLD);
}
