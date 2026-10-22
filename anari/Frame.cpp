// Copyright 2021-2026 The Khronos Group
// Copyright (c) 2026 Tachyon-dev contributors (adapted from the helide device)
// SPDX-License-Identifier: Apache-2.0

#include "Frame.h"
// std
#include <algorithm>
#include <chrono>
#include <cstring>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace tachyon_anari {

// Tachyon keeps process-global state (image tables, thread pools per scene);
// serialize all scene building and rendering.
static std::mutex &tachyonMutex()
{
  static std::mutex m;
  return m;
}

// Styles for material scalars (opacity, metallic, roughness) that come from
// per-vertex or per-primitive attributes: the values are quantized to 16
// levels and one style is created per distinct combination, so memory stays
// bounded by the number of combinations (at most 16^3), not the number of
// primitives.
namespace {
struct AttributeStyleBinder : public PrimStyleBinder
{
  static constexpr int kLevels = 15;

  TachyonScene *scene{nullptr};
  const Material *material{nullptr};
  SurfaceStyle baseStyle;
  float colorAlpha{1.f};
  Material::Scalars scalars{};
  int attr[3]{-1, -1, -1}; // opacity, metallic, roughness: attribute or -1
  std::unordered_map<unsigned, int> cache;

  static unsigned level(float v)
  {
    if (!(v > 0.f))
      return 0;
    return v >= 1.f ? kLevels : unsigned(std::lround(v * kLevels));
  }

  int styleFor(const float4 *vals) override
  {
    Material::Scalars s = scalars;
    float *dst[3] = {&s.opacity, &s.metallic, &s.roughness};
    unsigned key = 0;
    for (int i = 0; i < 3; i++) {
      if (attr[i] < 0)
        continue;
      const unsigned l = level(vals[attr[i]].x);
      key |= l << (5 * i);
      *dst[i] = float(l) / kLevels;
    }
    auto it = cache.find(key);
    if (it != cache.end())
      return it->second;
    SurfaceStyle st = baseStyle;
    const int id = material->fillStyle(st, colorAlpha, &s) ? scene->addStyle(st) : -1;
    cache.emplace(key, id);
    return id;
  }
};
} // namespace

// Row 0 of Tachyon's float output is the top image row; ANARI frame buffers
// start at the bottom row.
static constexpr bool kTachyonRowsTopDown = true;

Frame::Frame(TachyonGlobalState *s) : helium::BaseFrame(s) {}

bool Frame::isValid() const
{
  return m_renderer && m_renderer->isValid() && m_camera && m_camera->isValid()
      && m_world && m_world->isValid() && m_size.x > 0 && m_size.y > 0;
}

TachyonGlobalState *Frame::deviceState() const
{
  return static_cast<TachyonGlobalState *>(helium::BaseObject::m_state);
}

void Frame::commitParameters()
{
  m_renderer = getParamObject<Renderer>("renderer");
  m_camera = getParamObject<Camera>("camera");
  m_world = getParamObject<World>("world");
  m_incomingColorType =
      getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
  m_incomingDepthType =
      getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
  m_incomingSize = getParam<uint2>("size", uint2(0u));
}

void Frame::finalize()
{
  if (!m_renderer)
    reportMessage(ANARI_SEVERITY_WARNING,
        "missing required parameter 'renderer' on frame");
  if (!m_camera)
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
  if (!m_world)
    reportMessage(
        ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
  if (m_incomingSize.x == 0 || m_incomingSize.y == 0)
    reportMessage(ANARI_SEVERITY_WARNING, "invalid frame dimensions");

  m_size = m_incomingSize;
  m_colorType = m_incomingColorType;
  m_depthType = m_incomingDepthType;

  const size_t numPixels = size_t(m_size.x) * m_size.y;
  const size_t bytesPerPixel = m_colorType == ANARI_FLOAT32_VEC4 ? 16 : 4;
  m_pixelBuffer.assign(m_colorType == ANARI_UNKNOWN ? 0 : numPixels * bytesPerPixel, 0);
  m_depthBuffer.clear();
  if (m_depthType == ANARI_FLOAT32)
    m_depthBuffer.assign(numPixels, std::numeric_limits<float>::infinity());
  m_depthRaw.clear();
  if (m_depthType == ANARI_FLOAT32)
    m_depthRaw.assign(numPixels, 1e18f);
  m_rgb.assign(numPixels * 3, 0.f);
}

bool Frame::getProperty(const std::string_view &name,
    ANARIDataType type,
    void *ptr,
    uint64_t,
    uint32_t)
{
  if (type == ANARI_FLOAT32 && name == "duration") {
    helium::writeToVoidP(ptr, m_duration);
    return true;
  }
  return false;
}

void Frame::rebuildScene(helium::TimeStamp sceneStamp)
{
  auto *state = deviceState();

  std::vector<SurfaceInstance> surfaces;
  std::vector<LightInstance> lights;
  m_world->collect(surfaces, lights);

  m_lit = !lights.empty() && !m_renderer->eyeLight();
  m_builtEyeLight = m_renderer->eyeLight();
  m_scene.reset(m_lit, state->numThreads);

  if (m_lit) {
    for (auto &li : lights)
      li.light->emit(m_scene, li.xfm);
  }

  for (auto &si : surfaces) {
    const Geometry *g = si.surface->geometry();
    const Material *m = si.surface->material();

    SurfaceStyle style;
    std::unique_ptr<PrimStyleBinder> binder;
    int colorAttr = -1;
    float colorAlpha = 1.f;

    if (!m) { // like helide: show surfaces without a material in magenta
      style.color[0] = 1.f;
      style.color[1] = 0.f;
      style.color[2] = 1.f;
    } else {
      if (!m->isValid()) {
        reportMessage(ANARI_SEVERITY_DEBUG,
            "skipping surface with unsupported material");
        continue;
      }
      const int attr = m->colorAttribute();
      if (attr >= 0) {
        // lookup order: instance attribute, then geometry vertex, primitive
        // and uniform attributes
        const std::optional<float4> *uniform = nullptr;
        if (si.instAttr && (*si.instAttr)[attr])
          uniform = &(*si.instAttr)[attr];
        else {
          switch (g->attributeSource(attr)) {
          case ColorSource::VERTEX:
          case ColorSource::PRIMITIVE:
            colorAttr = attr;
            break;
          case ColorSource::UNIFORM:
            uniform = &g->uniformAttribute(attr);
            break;
          case ColorSource::NONE:
            break;
          }
        }
        if (uniform) {
          style.color[0] = (*uniform)->x;
          style.color[1] = (*uniform)->y;
          style.color[2] = (*uniform)->z;
          colorAlpha = (*uniform)->w;
        }
      }

      // opacity / metallic / roughness may also name an attribute; resolve
      // them like the color: instance, then vertex/primitive (per primitive,
      // below), then uniform. An attribute that is not set reads (0,0,0,1).
      Material::Scalars sc = m->scalars();
      // (the eyeLight preview ignores all three)
      const bool preview = m_renderer->eyeLight();
      const int sattr[3] = {preview ? -1 : m->opacityAttribute(),
          preview ? -1 : m->metallicAttribute(),
          preview ? -1 : m->roughnessAttribute()};
      float *sval[3] = {&sc.opacity, &sc.metallic, &sc.roughness};
      int perPrim[3] = {-1, -1, -1};
      unsigned mask = 0;
      for (int i = 0; i < 3; i++) {
        const int a = sattr[i];
        if (a < 0)
          continue;
        if (si.instAttr && (*si.instAttr)[a]) {
          *sval[i] = (*si.instAttr)[a]->x;
          continue;
        }
        switch (g->attributeSource(a)) {
        case ColorSource::VERTEX:
        case ColorSource::PRIMITIVE:
          perPrim[i] = a;
          mask |= 1u << a;
          break;
        case ColorSource::UNIFORM:
          *sval[i] = g->uniformAttribute(a)->x;
          break;
        case ColorSource::NONE:
          *sval[i] = 0.f;
          break;
        }
      }

      if (mask) {
        auto b = std::make_unique<AttributeStyleBinder>();
        b->scene = &m_scene;
        b->material = m;
        b->baseStyle = style;
        b->colorAlpha = colorAlpha;
        b->scalars = sc;
        for (int i = 0; i < 3; i++)
          b->attr[i] = perPrim[i];
        b->attrMask = mask;
        binder = std::move(b);
      } else if (!m->fillStyle(style, colorAlpha, &sc, preview)) {
        continue; // fully transparent
      }
    }

    if (binder) {
      g->emit(m_scene, -1, si.xfm, colorAttr, binder.get());
      continue;
    }
    const int id = m_scene.addStyle(style);
    g->emit(m_scene, id, si.xfm, colorAttr, nullptr);
  }

  m_scene.finishBuild();

  m_builtStamp = sceneStamp;
  m_builtWorld = m_world.ptr;
  m_sceneValid = true;
  m_shadingDirty = true;
  state->sceneBuilds++;
  reportMessage(ANARI_SEVERITY_DEBUG,
      "tachyon: rebuilt scene (%zu triangles, %zu spheres, %zu cylinders, "
      "%zu lights)",
      m_scene.numTriangles(),
      m_scene.numSpheres(),
      m_scene.numCylinders(),
      lights.size());
}

void Frame::fillWithBackground()
{
  const float4 bg = m_renderer ? m_renderer->background() : float4(0.f);
  m_rgb.assign(m_rgb.size(), 0.f);
  for (size_t i = 0; i + 2 < m_rgb.size(); i += 3) {
    m_rgb[i] = bg.x;
    m_rgb[i + 1] = bg.y;
    m_rgb[i + 2] = bg.z;
  }
  writePixels();
}

void Frame::renderFrame()
{
  auto *state = deviceState();
  const auto start = std::chrono::steady_clock::now();

  state->commitBuffer.flush();

  if (!isValid()) {
    if (m_size.x > 0 && m_size.y > 0) {
      reportMessage(
          ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
      fillWithBackground();
    }
    return;
  }

  std::lock_guard<std::mutex> lock(tachyonMutex());

  // Rebuild geometry + acceleration structure only when scene objects changed
  const helium::TimeStamp sceneStamp = state->sceneStamp.load();
  if (!m_sceneValid || m_builtWorld != m_world.ptr || sceneStamp > m_builtStamp
      || m_builtEyeLight != m_renderer->eyeLight())
    rebuildScene(sceneStamp);

  // Renderer changes only touch cheap shading weights
  if (m_shadingDirty || m_shadedWith != m_renderer.ptr
      || m_shadedStamp != m_renderer->lastFinalized()) {
    float ambient, diffuse;
    m_renderer->shadingWeights(m_lit, ambient, diffuse);
    m_scene.setShading(ambient, diffuse);
    m_shadedWith = m_renderer.ptr;
    m_shadedStamp = m_renderer->lastFinalized();
    m_shadingDirty = false;
  }

  const float frameAspect = float(m_size.x) / float(m_size.y);
  SceneCamera cam;
  m_camera->fill(cam, frameAspect);

  m_scene.setOutput(int(m_size.x), int(m_size.y), m_rgb.data());
  m_scene.setCamera(cam);
  m_scene.setDepthOutput(m_depthRaw.empty() ? nullptr : m_depthRaw.data());
  const float4 &bg = m_renderer->background();
  m_scene.setBackground(&bg.x);
  m_scene.setSamples(m_renderer->pixelSamples());

  m_scene.render();
  writePixels();

  m_duration = std::chrono::duration<float>(
      std::chrono::steady_clock::now() - start)
                   .count();
}

void Frame::writePixels()
{
  // Depth: distance along the (normalized) primary ray, +inf for misses
  if (!m_depthBuffer.empty() && m_depthRaw.size() == m_depthBuffer.size()) {
    const size_t dw = m_size.x, dh = m_size.y;
    for (size_t y = 0; y < dh; y++) {
      const size_t srcRow = kTachyonRowsTopDown ? dh - 1 - y : y;
      const float *src = m_depthRaw.data() + srcRow * dw;
      float *dst = m_depthBuffer.data() + y * dw;
      for (size_t x = 0; x < dw; x++)
        dst[x] = src[x] >= 1e17f ? std::numeric_limits<float>::infinity()
                                 : src[x];
    }
  }

  if (m_colorType == ANARI_UNKNOWN)
    return;

  const float4 bg = m_renderer ? m_renderer->background() : float4(0.f);
  const bool bgOpaque = bg.w >= 1.f;
  const size_t w = m_size.x, h = m_size.y;

  for (size_t y = 0; y < h; y++) {
    const size_t srcRow = kTachyonRowsTopDown ? h - 1 - y : y;
    const float *src = m_rgb.data() + srcRow * w * 3;
    uint8_t *dst = m_pixelBuffer.data()
        + y * w * (m_colorType == ANARI_FLOAT32_VEC4 ? 16 : 4);
    for (size_t x = 0; x < w; x++) {
      const float r = src[3 * x + 0], g = src[3 * x + 1], b = src[3 * x + 2];
      // Tachyon has no alpha channel: pixels equal to the background color
      // are treated as background, everything else as covered.
      float a = 1.f;
      if (!bgOpaque && r == bg.x && g == bg.y && b == bg.z)
        a = bg.w;
      const float4 c(std::min(r, 1.f), std::min(g, 1.f), std::min(b, 1.f), a);

      switch (m_colorType) {
      case ANARI_UFIXED8_VEC4: {
        const uint32_t v = helium::math::cvt_color_to_uint32(c);
        std::memcpy(dst + 4 * x, &v, 4);
        break;
      }
      case ANARI_UFIXED8_RGBA_SRGB: {
        const uint32_t v = helium::math::cvt_color_to_uint32_srgb(c);
        std::memcpy(dst + 4 * x, &v, 4);
        break;
      }
      case ANARI_FLOAT32_VEC4:
        std::memcpy(dst + 16 * x, &c, 16);
        break;
      default:
        break;
      }
    }
  }
  // Depth: Tachyon's API exposes no primary-hit distance, so the channel stays
  // at +inf (set in finalize()).
}

void *Frame::map(std::string_view channel,
    uint32_t *width,
    uint32_t *height,
    ANARIDataType *pixelType)
{
  *width = m_size.x;
  *height = m_size.y;

  if (channel == "channel.color" && !m_pixelBuffer.empty()) {
    *pixelType = m_colorType;
    return m_pixelBuffer.data();
  }
  if (channel == "channel.depth" && !m_depthBuffer.empty()) {
    *pixelType = m_depthType;
    return m_depthBuffer.data();
  }
  *width = 0;
  *height = 0;
  *pixelType = ANARI_UNKNOWN;
  return nullptr;
}

void Frame::unmap(std::string_view) {}

int Frame::frameReady(ANARIWaitMask)
{
  return 1; // rendering is synchronous
}

void Frame::discard() {}

} // namespace tachyon_anari

namespace anari {
ANARI_TYPEFOR_DEFINITION(tachyon_anari::Frame *);
}
