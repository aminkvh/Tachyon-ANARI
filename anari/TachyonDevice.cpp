// Copyright 2021-2026 The Khronos Group
// Copyright (c) 2026 Tachyon-dev contributors (adapted from the helide device)
// SPDX-License-Identifier: Apache-2.0

#include "TachyonDevice.h"

#include "Camera.h"
#include "Frame.h"
#include "Geometry.h"
#include "Light.h"
#include "Material.h"
#include "Renderer.h"
#include "World.h"
// helium
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"
#include "helium/array/Array3D.h"
#include "helium/array/ObjectArray.h"
// std
#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>

#include "anari_library_tachyon_queries.h"

namespace tachyon_anari {

// Object types the device does not implement. They are kept (so applications
// can set parameters and release them as usual) but never rendered.
struct UnsupportedObject : public Object
{
  UnsupportedObject(ANARIDataType type, TachyonGlobalState *s) : Object(type, s)
  {}
  bool isValid() const override { return false; }
};

template <typename T>
static T *warnUnsupported(T *obj, const char *kind, const char *subtype)
{
  obj->reportMessage(ANARI_SEVERITY_WARNING,
      "tachyon: %s '%s' is not supported; objects using it will not be rendered",
      kind,
      subtype ? subtype : "");
  return obj;
}

// Data Arrays ////////////////////////////////////////////////////////////////

ANARIArray1D TachyonDevice::newArray1D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems)
{
  initDevice();

  helium::Array1DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems = numItems;

  if (anari::isObject(type))
    return (ANARIArray1D) new helium::ObjectArray(deviceState(), md);
  else
    return (ANARIArray1D) new helium::Array1D(deviceState(), md);
}

ANARIArray2D TachyonDevice::newArray2D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2)
{
  initDevice();

  helium::Array2DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;

  return (ANARIArray2D) new helium::Array2D(deviceState(), md);
}

ANARIArray3D TachyonDevice::newArray3D(const void *appMemory,
    ANARIMemoryDeleter deleter,
    const void *userData,
    ANARIDataType type,
    uint64_t numItems1,
    uint64_t numItems2,
    uint64_t numItems3)
{
  initDevice();

  helium::Array3DMemoryDescriptor md;
  md.appMemory = appMemory;
  md.deleter = deleter;
  md.deleterPtr = userData;
  md.elementType = type;
  md.numItems1 = numItems1;
  md.numItems2 = numItems2;
  md.numItems3 = numItems3;

  return (ANARIArray3D) new helium::Array3D(deviceState(), md);
}

ANARICamera TachyonDevice::newCamera(const char *subtype)
{
  initDevice();
  if (auto *c = Camera::createInstance(subtype, deviceState()))
    return (ANARICamera)c;
  return (ANARICamera)warnUnsupported(
      new UnsupportedCamera(deviceState()), "camera", subtype);
}

ANARIFrame TachyonDevice::newFrame()
{
  initDevice();
  return (ANARIFrame) new Frame(deviceState());
}

ANARIGeometry TachyonDevice::newGeometry(const char *subtype)
{
  initDevice();
  if (auto *g = Geometry::createInstance(subtype, deviceState()))
    return (ANARIGeometry)g;
  return (ANARIGeometry)warnUnsupported(
      new UnsupportedGeometry(deviceState()), "geometry", subtype);
}

ANARIGroup TachyonDevice::newGroup()
{
  initDevice();
  return (ANARIGroup) new Group(deviceState());
}

ANARIInstance TachyonDevice::newInstance(const char *subtype)
{
  initDevice();
  auto *i = new Instance(deviceState());
  if (subtype && std::string(subtype) != "transform") {
    i->reportMessage(ANARI_SEVERITY_WARNING,
        "tachyon: instance type '%s' is not supported, treating it as "
        "'transform'",
        subtype);
  }
  return (ANARIInstance)i;
}

ANARILight TachyonDevice::newLight(const char *subtype)
{
  initDevice();
  if (auto *l = Light::createInstance(subtype, deviceState()))
    return (ANARILight)l;
  return (ANARILight)warnUnsupported(
      new UnsupportedLight(deviceState()), "light", subtype);
}

ANARIMaterial TachyonDevice::newMaterial(const char *subtype)
{
  initDevice();
  if (auto *m = Material::createInstance(subtype, deviceState()))
    return (ANARIMaterial)m;
  return (ANARIMaterial)warnUnsupported(
      new UnsupportedMaterial(deviceState()), "material", subtype);
}

ANARIRenderer TachyonDevice::newRenderer(const char *subtype)
{
  initDevice();
  auto *r = new Renderer(deviceState());
  if (subtype && std::string(subtype) != "default") {
    r->reportMessage(ANARI_SEVERITY_WARNING,
        "tachyon: renderer '%s' is not supported, using 'default'",
        subtype);
  }
  return (ANARIRenderer)r;
}

ANARISampler TachyonDevice::newSampler(const char *subtype)
{
  initDevice();
  return (ANARISampler)warnUnsupported(
      new UnsupportedObject(ANARI_SAMPLER, deviceState()), "sampler", subtype);
}

ANARISpatialField TachyonDevice::newSpatialField(const char *subtype)
{
  initDevice();
  return (ANARISpatialField)warnUnsupported(
      new UnsupportedObject(ANARI_SPATIAL_FIELD, deviceState()),
      "spatial field",
      subtype);
}

ANARISurface TachyonDevice::newSurface()
{
  initDevice();
  return (ANARISurface) new Surface(deviceState());
}

ANARIVolume TachyonDevice::newVolume(const char *subtype)
{
  initDevice();
  return (ANARIVolume)warnUnsupported(
      new UnsupportedObject(ANARI_VOLUME, deviceState()), "volume", subtype);
}

ANARIWorld TachyonDevice::newWorld()
{
  initDevice();
  return (ANARIWorld) new World(deviceState());
}

// Query functions ////////////////////////////////////////////////////////////

const char **TachyonDevice::getObjectSubtypes(ANARIDataType objectType)
{
  return tachyon_anari::query_object_types(objectType);
}

const void *TachyonDevice::getObjectInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *infoName,
    ANARIDataType infoType)
{
  return tachyon_anari::query_object_info(
      objectType, objectSubtype, infoName, infoType);
}

const void *TachyonDevice::getParameterInfo(ANARIDataType objectType,
    const char *objectSubtype,
    const char *parameterName,
    ANARIDataType parameterType,
    const char *infoName,
    ANARIDataType infoType)
{
  return tachyon_anari::query_param_info(objectType,
      objectSubtype,
      parameterName,
      parameterType,
      infoName,
      infoType);
}

// Other TachyonDevice definitions ////////////////////////////////////////////

TachyonDevice::TachyonDevice(ANARIStatusCallback cb, const void *ptr)
    : helium::BaseDevice(cb, ptr)
{
  m_state = std::make_unique<TachyonGlobalState>(this_device());
  deviceCommitParameters();
}

TachyonDevice::TachyonDevice(ANARILibrary l) : helium::BaseDevice(l)
{
  m_state = std::make_unique<TachyonGlobalState>(this_device());
  deviceCommitParameters();
}

TachyonDevice::~TachyonDevice()
{
  auto &state = *deviceState();
  state.commitBuffer.clear();
  reportMessage(ANARI_SEVERITY_DEBUG, "destroying tachyon device (%p)", this);
}

void TachyonDevice::initDevice()
{
  if (m_initialized)
    return;

  reportMessage(ANARI_SEVERITY_DEBUG, "initializing tachyon device (%p)", this);

  auto &state = *deviceState();
  state.anariDevice = (anari::Device)this;

  // Default: one render thread per CPU (Tachyon picks it when <= 0)
  state.numThreads = -1;
  if (const char *env = std::getenv("TACHYON_NUM_THREADS"))
    state.numThreads = std::atoi(env);

  m_initialized = true;
}

void TachyonDevice::deviceCommitParameters()
{
  helium::BaseDevice::deviceCommitParameters();
}

#define TACHYON_STRINGIFY(s) TACHYON_STRINGIFY2(s)
#define TACHYON_STRINGIFY2(s) #s
#define TACHYON_VERSION_STRING                                                 \
  "tachyon " TACHYON_STRINGIFY(ANARI_SDK_VERSION_MAJOR) "." TACHYON_STRINGIFY( \
      ANARI_SDK_VERSION_MINOR) "." TACHYON_STRINGIFY(ANARI_SDK_VERSION_PATCH)

int TachyonDevice::deviceGetProperty(const char *name,
    ANARIDataType type,
    void *mem,
    uint64_t size,
    uint32_t)
{
  static const std::string version = TACHYON_VERSION_STRING;
  std::string_view prop = name;
  if (prop == "extension" && type == ANARI_STRING_LIST) {
    helium::writeToVoidP(mem, query_extensions());
    return 1;
  } else if (prop == "version" && type == ANARI_INT32) {
    int v = ANARI_SDK_VERSION_MAJOR * 1000 + ANARI_SDK_VERSION_MINOR * 100
        + ANARI_SDK_VERSION_PATCH;
    helium::writeToVoidP(mem, v);
    return 1;
  } else if (prop == "version.major" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, int(ANARI_SDK_VERSION_MAJOR));
    return 1;
  } else if (prop == "version.minor" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, int(ANARI_SDK_VERSION_MINOR));
    return 1;
  } else if (prop == "version.patch" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, int(ANARI_SDK_VERSION_PATCH));
    return 1;
  } else if (prop == "version.name.size" && type == ANARI_UINT64) {
    helium::writeToVoidP(mem, uint64_t(version.size() + 1));
    return 1;
  } else if (prop == "version.name" && type == ANARI_STRING) {
    std::memset(mem, 0, size);
    std::memcpy(mem,
        version.data(),
        std::min(uint64_t(version.size()), size - 1));
    return 1;
  } else if (prop == "anariVersion.major" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, 1);
    return 1;
  } else if (prop == "anariVersion.minor" && type == ANARI_INT32) {
    helium::writeToVoidP(mem, 1);
    return 1;
  } else if (prop == "geometryMaxIndex" && type == ANARI_UINT64) {
    helium::writeToVoidP(mem, uint64_t(std::numeric_limits<uint32_t>::max()));
    return 1;
  } else if (prop == "tachyon" && type == ANARI_BOOL) {
    helium::writeToVoidP(mem, true);
    return 1;
  } else if (prop == "tachyon.sceneBuilds" && type == ANARI_UINT64) {
    // diagnostic: number of Tachyon scene (geometry + BVH) rebuilds so far
    helium::writeToVoidP(mem, uint64_t(deviceState()->sceneBuilds.load()));
    return 1;
  }
  return 0;
}

TachyonGlobalState *TachyonDevice::deviceState() const
{
  return static_cast<TachyonGlobalState *>(helium::BaseDevice::m_state.get());
}

} // namespace tachyon_anari
