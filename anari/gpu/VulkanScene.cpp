// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "VulkanScene.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace tachyon_anari {
namespace gpu {

namespace {

// Matches FrameData in shaders/raytrace.slang
struct FrameData
{
  float posOrtho[4];
  float fwdU0[4];
  float rightDu[4];
  float upV0[4];
  float shade[4];
  float bg[4];
  float unused0[4];
  uint32_t size[4];
  uint32_t counts[4];
  uint32_t flags[4];
  uint32_t styles[4];
};
static_assert(sizeof(FrameData) == 176, "uniform block size");

enum : uint32_t
{
  F_SPHERE_COLORS = 1,
  F_SPHERE_STYLES = 2,
  F_CYL_COLORS = 4,
  F_CYL_STYLES = 8,
  F_SHADOWS = 16,
  F_DEPTH = 32,
  F_ACCUM = 64,
  F_ACCUM_READ = 128,
  F_BG_ALPHA = 256,
  F_TRANSPARENT = 512
};

enum : unsigned
{
  WARN_SAMPLER = 1,
  WARN_SAMPLES = 2
};

enum
{
  BIND_AS = 0,
  BIND_SPHERES = 1,
  BIND_SPHERE_COLORS = 2,
  BIND_SPHERE_STYLES = 3,
  BIND_CYLS = 4,
  BIND_CYL_COLORS = 5,
  BIND_CYL_STYLES = 6,
  BIND_TRI_POS = 7,
  BIND_TRI_SHADE = 8,
  BIND_TRI_AUX = 9,
  BIND_STYLES = 10,
  BIND_SAMPLERS = 11,
  BIND_TEXELS = 12,
  BIND_LIGHTS = 13,
  BIND_OUT = 14,
  BIND_DEPTH = 15,
  BIND_ACCUM = 16,
  BIND_AABBS = 17,
  BIND_INSTANCES = 18,
  BIND_FRAME = 19,
  NUM_BINDINGS = 20
};
static_assert(NUM_BINDINGS == kNumBindings, "descriptor layout");

double nowMs()
{
  using namespace std::chrono;
  return duration<double, std::milli>(steady_clock::now().time_since_epoch())
      .count();
}

void memoryBarrier(VkCommandBuffer cb,
    VkPipelineStageFlags srcStage,
    VkAccessFlags srcAccess,
    VkPipelineStageFlags dstStage,
    VkAccessFlags dstAccess)
{
  VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
  b.srcAccessMask = srcAccess;
  b.dstAccessMask = dstAccess;
  vkCmdPipelineBarrier(cb, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

void check(VkResult r, const char *what)
{
  if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY)
    throw std::runtime_error(std::string(what) + ": out of memory");
  if (r != VK_SUCCESS)
    throw std::runtime_error(
        std::string(what) + " failed (" + std::to_string(int(r)) + ")");
}

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

std::unique_ptr<VulkanScene> VulkanScene::create(
    std::string &gpuName, std::string &reason, bool &hard)
{
  auto ctx = VulkanContext::acquire(reason, hard);
  if (!ctx)
    return nullptr;
  gpuName = ctx->gpuName();
  return std::unique_ptr<VulkanScene>(new VulkanScene(std::move(ctx)));
}

VulkanScene::VulkanScene(std::shared_ptr<VulkanContext> ctx)
    : m_ctx(std::move(ctx))
{
  const char *dbg = std::getenv("TACHYON_VULKAN_DEBUG");
  m_debug = dbg && *dbg && *dbg != '0';
  const char *acc = std::getenv("TACHYON_VULKAN_ACCUM");
  m_accumEnabled = !(acc && *acc == '0');
  if (const char *r = std::getenv("TACHYON_VULKAN_REFITS"))
    m_maxRefits = unsigned(std::max(0, std::atoi(r)));
  m_protos.emplace_back();
  createPipelines();
}

VulkanScene::~VulkanScene()
{
  VkDevice dev = m_ctx->device();
  vkDeviceWaitIdle(dev);
  destroyScene();
  destroyOutputs();
  m_ctx->destroyBuffer(m_dummy);
  m_ctx->destroyBuffer(m_frameUbo);
  m_ctx->destroyBuffer(m_lightBuf);
  if (m_descPool)
    vkDestroyDescriptorPool(dev, m_descPool, nullptr);
}

void VulkanScene::createPipelines()
{
  VkDevice dev = m_ctx->device();
  VkDescriptorPoolSize sizes[3] = {
      {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NUM_BINDINGS - 2},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1}};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 1;
  dpi.poolSizeCount = 3;
  dpi.pPoolSizes = sizes;
  check(vkCreateDescriptorPool(dev, &dpi, nullptr, &m_descPool),
      "vkCreateDescriptorPool");
  VkDescriptorSetLayout layout = m_ctx->setLayout();
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = m_descPool;
  dai.descriptorSetCount = 1;
  dai.pSetLayouts = &layout;
  check(vkAllocateDescriptorSets(dev, &dai, &m_set), "vkAllocateDescriptorSets");

  m_dummy = m_ctx->createBuffer(
      64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::DeviceLocal);
  m_frameUbo = m_ctx->createBuffer(
      sizeof(FrameData), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, MemoryKind::Upload);
}

void VulkanScene::destroyBlas(Blas &b)
{
  if (b.as)
    vkDestroyAccelerationStructureKHR(m_ctx->device(), b.as, nullptr);
  m_ctx->destroyBuffer(b.buffer);
  b = Blas();
}

void VulkanScene::destroyScene()
{
  VkDevice dev = m_ctx->device();
  if (m_tlas)
    vkDestroyAccelerationStructureKHR(dev, m_tlas, nullptr);
  m_tlas = VK_NULL_HANDLE;
  m_ctx->destroyBuffer(m_tlasBuf);
  for (auto &b : m_blas)
    destroyBlas(b);
  m_blas.clear();
  for (auto &b : m_buf)
    m_ctx->destroyBuffer(b);
  m_ctx->destroyBuffer(m_lightBuf);
  m_lightCapacity = 0;
  m_haveSig = false;
  m_updatable = false;
  m_refits = 0;
  m_descDirty = true;
}

void VulkanScene::destroyOutputs()
{
  m_ctx->destroyBuffer(m_outBuf);
  m_ctx->destroyBuffer(m_depthBuf);
  m_ctx->destroyBuffer(m_readBuf);
  m_ctx->destroyBuffer(m_depthReadBuf);
  m_ctx->destroyBuffer(m_accumBuf);
  m_outW = m_outH = 0;
  m_outBytes = 0;
  m_descDirty = true;
}

void VulkanScene::warnOnce(unsigned id, const char *text)
{
  if (m_warned & id)
    return;
  m_warned |= id;
  m_warnings.emplace_back(text);
}

std::vector<std::string> VulkanScene::takeWarnings()
{
  std::vector<std::string> w;
  w.swap(m_warnings);
  return w;
}

// Scene description ///////////////////////////////////////////////////////////

// The GPU scene of the previous build stays alive: finishBuild() decides
// whether it can be refitted.
void VulkanScene::reset(bool lit, int)
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

int VulkanScene::addStyle(const SurfaceStyle &style)
{
  m_styles.push_back(style);
  if (style.opacity < 1.f)
    m_transparent = true;
  return int(m_styles.size()) - 1;
}

void VulkanScene::addDirectionalLight(const float *direction, const float *rgb)
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

void VulkanScene::addPointLight(const float *position, const float *rgb)
{
  Light l{};
  for (int i = 0; i < 3; i++) {
    l.v[i] = position[i];
    l.color[i] = rgb[i];
  }
  l.type = 1;
  m_lights.push_back(l);
}

int VulkanScene::beginPrototype()
{
  m_protos.emplace_back();
  m_cur = m_protos.size() - 1;
  return int(m_cur);
}

void VulkanScene::endPrototype()
{
  m_cur = 0;
}

void VulkanScene::addInstance(int prototype, const float *x)
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

void VulkanScene::addTriangle(
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

void VulkanScene::addSampledTriangle(int style,
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
void VulkanScene::ensureColors(Proto &p, bool sphere)
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

void VulkanScene::ensureStyles(Proto &p, bool sphere)
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

void VulkanScene::addSphere(
    int style, const float *center, float radius, const float *rgb)
{
  addSpheres(style, 1, center, &radius, radius, rgb);
}

void VulkanScene::addSpheres(int style,
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

void VulkanScene::addCylinder(int style,
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

// Build ///////////////////////////////////////////////////////////////////////

void VulkanScene::upload(GpuBuffer &dst, const void *src, size_t bytes)
{
  const size_t chunk = size_t(8) << 20;
  GpuBuffer stage = m_ctx->createBuffer(std::min(bytes, chunk),
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      MemoryKind::Upload);
  try {
    for (size_t off = 0; off < bytes; off += chunk) {
      const size_t n = std::min(chunk, bytes - off);
      std::memcpy(stage.mapped, static_cast<const char *>(src) + off, n);
      m_ctx->syncMapped(stage, true);
      m_ctx->submitAndWait([&](VkCommandBuffer cb) {
        VkBufferCopy region{0, off, n};
        vkCmdCopyBuffer(cb, stage.buffer, dst.buffer, 1, &region);
      });
    }
  } catch (...) {
    m_ctx->destroyBuffer(stage);
    throw;
  }
  m_ctx->destroyBuffer(stage);
}

GpuBuffer VulkanScene::deviceBuffer(
    const void *data, size_t bytes, VkBufferUsageFlags usage)
{
  GpuBuffer b = m_ctx->createBuffer(
      bytes, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::DeviceLocal);
  try {
    upload(b, data, bytes);
  } catch (...) {
    m_ctx->destroyBuffer(b);
    throw;
  }
  return b;
}

void VulkanScene::putBuffer(
    int binding, const void *data, size_t bytes, VkBufferUsageFlags usage)
{
  if (m_buf[binding])
    upload(m_buf[binding], data, bytes);
  else
    m_buf[binding] = deviceBuffer(data, bytes, usage);
}

void VulkanScene::updateDescriptors()
{
  VkDevice dev = m_ctx->device();
  const GpuBuffer *src[NUM_BINDINGS] = {};
  for (int i = BIND_SPHERES; i < NUM_BINDINGS; i++)
    src[i] = &m_buf[i];
  src[BIND_LIGHTS] = &m_lightBuf;
  src[BIND_OUT] = &m_outBuf;
  src[BIND_DEPTH] = &m_depthBuf;
  src[BIND_ACCUM] = &m_accumBuf;
  src[BIND_FRAME] = &m_frameUbo;

  VkDescriptorBufferInfo infos[NUM_BINDINGS];
  VkWriteDescriptorSet w[NUM_BINDINGS]{};
  uint32_t n = 0;
  VkWriteDescriptorSetAccelerationStructureKHR asw{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  if (m_tlas) {
    asw.accelerationStructureCount = 1;
    asw.pAccelerationStructures = &m_tlas;
    w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[n].pNext = &asw;
    w[n].dstSet = m_set;
    w[n].dstBinding = BIND_AS;
    w[n].descriptorCount = 1;
    w[n].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    n++;
  }
  for (int i = BIND_SPHERES; i < NUM_BINDINGS; i++) {
    const GpuBuffer &b = (src[i] && *src[i]) ? *src[i] : m_dummy;
    infos[i] = {b.buffer, 0, VK_WHOLE_SIZE};
    w[n].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[n].dstSet = m_set;
    w[n].dstBinding = uint32_t(i);
    w[n].descriptorCount = 1;
    w[n].descriptorType = i == BIND_FRAME ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                          : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[n].pBufferInfo = &infos[i];
    n++;
  }
  vkUpdateDescriptorSets(dev, n, w, 0, nullptr);
  m_descDirty = false;
}

void VulkanScene::generateAabbs(uint32_t set, uint32_t count, GpuBuffer &aabbs)
{
  m_buf[BIND_AABBS] = aabbs;
  updateDescriptors();
  FrameData fd{};
  fd.counts[0] = set == 0 ? count : 0;
  fd.counts[1] = set == 1 ? count : 0;
  fd.flags[3] = set;
  // 2D dispatch: one dimension is limited to 65535 groups
  const uint32_t groups = (count + 63) / 64;
  const uint32_t gx = std::min<uint32_t>(groups, 65535);
  const uint32_t gy = (groups + gx - 1) / gx;
  fd.styles[2] = gx * 64; // threads per row
  std::memcpy(m_frameUbo.mapped, &fd, sizeof(fd));
  m_ctx->syncMapped(m_frameUbo, true);
  m_ctx->submitAndWait([&](VkCommandBuffer cb) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->aabbPipeline());
    vkCmdBindDescriptorSets(cb,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        m_ctx->pipelineLayout(),
        0,
        1,
        &m_set,
        0,
        nullptr);
    vkCmdDispatch(cb, gx, gy, 1);
    memoryBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
        VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);
  });
  m_buf[BIND_AABBS] = GpuBuffer(); // owned by the caller
  m_descDirty = true;
}

VulkanScene::Blas VulkanScene::buildBlas(
    const VkAccelerationStructureGeometryKHR &geomIn,
    uint32_t prims,
    bool updatable)
{
  VkDevice dev = m_ctx->device();
  VkAccelerationStructureGeometryKHR geom = geomIn;
  VkAccelerationStructureBuildGeometryInfoKHR bgi{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
      | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR
      | (updatable ? VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR : 0);
  bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
  bgi.geometryCount = 1;
  bgi.pGeometries = &geom;
  VkAccelerationStructureBuildSizesInfoKHR sz{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
  vkGetAccelerationStructureBuildSizesKHR(
      dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bgi, &prims, &sz);

  const VkDeviceSize align = m_ctx->scratchAlignment();
  GpuBuffer scratch, big;
  VkAccelerationStructureKHR bigAs = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  Blas out;
  try {
    scratch = m_ctx->createBuffer(sz.buildScratchSize + align,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        MemoryKind::DeviceLocal);
    bgi.scratchData.deviceAddress = (scratch.address + align - 1) / align * align;
    big = m_ctx->createBuffer(sz.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        MemoryKind::DeviceLocal);
    VkAccelerationStructureCreateInfoKHR asci{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    asci.buffer = big.buffer;
    asci.size = sz.accelerationStructureSize;
    asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    check(vkCreateAccelerationStructureKHR(dev, &asci, nullptr, &bigAs),
        "vkCreateAccelerationStructureKHR");
    bgi.dstAccelerationStructure = bigAs;

    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    qpi.queryCount = 1;
    check(vkCreateQueryPool(dev, &qpi, nullptr, &queries), "vkCreateQueryPool");

    VkAccelerationStructureBuildRangeInfoKHR range{prims, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
    m_ctx->submitAndWait([&](VkCommandBuffer cb) {
      vkCmdResetQueryPool(cb, queries, 0, 1);
      vkCmdBuildAccelerationStructuresKHR(cb, 1, &bgi, &pRange);
      memoryBarrier(cb,
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
      vkCmdWriteAccelerationStructuresPropertiesKHR(cb,
          1,
          &bigAs,
          VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
          queries,
          0);
    });
    VkDeviceSize compact = 0;
    check(vkGetQueryPoolResults(dev,
              queries,
              0,
              1,
              sizeof(compact),
              &compact,
              sizeof(compact),
              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "vkGetQueryPoolResults");
    if (compact == 0 || compact > sz.accelerationStructureSize)
      compact = sz.accelerationStructureSize;
    out.buffer = m_ctx->createBuffer(compact,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        MemoryKind::DeviceLocal);
    asci.buffer = out.buffer.buffer;
    asci.size = compact;
    check(vkCreateAccelerationStructureKHR(dev, &asci, nullptr, &out.as),
        "vkCreateAccelerationStructureKHR");
    m_ctx->submitAndWait([&](VkCommandBuffer cb) {
      VkCopyAccelerationStructureInfoKHR ci{
          VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
      ci.src = bigAs;
      ci.dst = out.as;
      ci.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
      vkCmdCopyAccelerationStructureKHR(cb, &ci);
    });
    VkAccelerationStructureDeviceAddressInfoKHR dai{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    dai.accelerationStructure = out.as;
    out.address = vkGetAccelerationStructureDeviceAddressKHR(dev, &dai);
    out.updateScratch = sz.updateScratchSize;
    if (m_debug)
      std::fprintf(stderr,
          "[tachyon vulkan]   BLAS %u prims: %.1f -> %.1f MB%s\n",
          prims,
          double(sz.accelerationStructureSize) / 1048576.0,
          double(compact) / 1048576.0,
          updatable ? " (updatable)" : "");
  } catch (...) {
    vkDeviceWaitIdle(dev);
    if (bigAs)
      vkDestroyAccelerationStructureKHR(dev, bigAs, nullptr);
    if (queries)
      vkDestroyQueryPool(dev, queries, nullptr);
    m_ctx->destroyBuffer(scratch);
    m_ctx->destroyBuffer(big);
    destroyBlas(out);
    throw;
  }
  vkDestroyAccelerationStructureKHR(dev, bigAs, nullptr);
  vkDestroyQueryPool(dev, queries, nullptr);
  m_ctx->destroyBuffer(scratch);
  m_ctx->destroyBuffer(big);
  return out;
}

// Refit: rebuild the tree bottom up for new primitive positions, keeping its
// topology (fast, the quality degrades with large motion)
void VulkanScene::updateBlas(Blas &blas,
    const VkAccelerationStructureGeometryKHR &geomIn,
    uint32_t prims)
{
  VkAccelerationStructureGeometryKHR geom = geomIn;
  VkAccelerationStructureBuildGeometryInfoKHR bgi{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
  bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
  bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
      | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR
      | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
  bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
  bgi.srcAccelerationStructure = blas.as;
  bgi.dstAccelerationStructure = blas.as;
  bgi.geometryCount = 1;
  bgi.pGeometries = &geom;
  const VkDeviceSize align = m_ctx->scratchAlignment();
  GpuBuffer scratch = m_ctx->createBuffer(blas.updateScratch + align,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
      MemoryKind::DeviceLocal);
  bgi.scratchData.deviceAddress = (scratch.address + align - 1) / align * align;
  VkAccelerationStructureBuildRangeInfoKHR range{prims, 0, 0, 0};
  const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
  try {
    m_ctx->submitAndWait([&](VkCommandBuffer cb) {
      vkCmdBuildAccelerationStructuresKHR(cb, 1, &bgi, &pRange);
      memoryBarrier(cb,
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    });
  } catch (...) {
    m_ctx->destroyBuffer(scratch);
    throw;
  }
  m_ctx->destroyBuffer(scratch);
}

void VulkanScene::buildTlas(const Gathered &g)
{
  VkDevice dev = m_ctx->device();
  if (m_tlas)
    vkDestroyAccelerationStructureKHR(dev, m_tlas, nullptr);
  m_tlas = VK_NULL_HANDLE;
  m_ctx->destroyBuffer(m_tlasBuf);

  const uint32_t count = uint32_t(g.instances.size());
  std::vector<VkAccelerationStructureInstanceKHR> inst(count);
  for (uint32_t i = 0; i < count; i++) {
    const InstanceEntry &e = g.instances[i];
    VkAccelerationStructureInstanceKHR &o = inst[i];
    std::memset(&o, 0, sizeof(o));
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 4; c++)
        o.transform.matrix[r][c] = e.m[4 * r + c];
    o.instanceCustomIndex = i; // index of the record in the instance table
    o.mask = 0xFF;
    o.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    o.accelerationStructureReference = m_blas[size_t(e.slice)].address;
  }
  const VkDeviceSize align = m_ctx->scratchAlignment();
  GpuBuffer instances = m_ctx->createBuffer(sizeof(inst[0]) * count,
      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      MemoryKind::Upload);
  GpuBuffer scratch;
  try {
    std::memcpy(instances.mapped, inst.data(), sizeof(inst[0]) * count);
    m_ctx->syncMapped(instances, true);

    VkAccelerationStructureGeometryKHR tg{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tg.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tg.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tg.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tg.geometry.instances.data.deviceAddress = instances.address;
    VkAccelerationStructureBuildGeometryInfoKHR tbi{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    tbi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    tbi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    tbi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    tbi.geometryCount = 1;
    tbi.pGeometries = &tg;
    VkAccelerationStructureBuildSizesInfoKHR tsz{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(
        dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tbi, &count, &tsz);
    scratch = m_ctx->createBuffer(tsz.buildScratchSize + align,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        MemoryKind::DeviceLocal);
    tbi.scratchData.deviceAddress = (scratch.address + align - 1) / align * align;
    m_tlasBuf = m_ctx->createBuffer(tsz.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        MemoryKind::DeviceLocal);
    VkAccelerationStructureCreateInfoKHR tci{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    tci.buffer = m_tlasBuf.buffer;
    tci.size = tsz.accelerationStructureSize;
    tci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    check(vkCreateAccelerationStructureKHR(dev, &tci, nullptr, &m_tlas),
        "vkCreateAccelerationStructureKHR");
    tbi.dstAccelerationStructure = m_tlas;
    VkAccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
    m_ctx->submitAndWait([&](VkCommandBuffer cb) {
      vkCmdBuildAccelerationStructuresKHR(cb, 1, &tbi, &pRange);
      memoryBarrier(cb,
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    });
  } catch (...) {
    m_ctx->destroyBuffer(scratch);
    m_ctx->destroyBuffer(instances);
    throw;
  }
  m_ctx->destroyBuffer(scratch);
  m_ctx->destroyBuffer(instances);
  m_descDirty = true;
}

// Brings the host description into its final layout: the arrays of all
// prototypes one after the other, one BLAS "slice" per prototype and type, one
// instance entry per placement and slice.
VulkanScene::Gathered VulkanScene::gather()
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

void VulkanScene::uploadSceneData(const Gathered &g)
{
  const VkBufferUsageFlags ssbo = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  putBuffer(BIND_STYLES, g.styles.data(), 4 * g.styles.size(), ssbo);
  putBuffer(BIND_INSTANCES, g.instTable.data(), 4 * g.instTable.size(), ssbo);
  if (!g.samplerTable.empty()) {
    putBuffer(BIND_SAMPLERS, g.samplerTable.data(), 4 * g.samplerTable.size(), ssbo);
    const float zero[4] = {0, 0, 0, 0};
    putBuffer(BIND_TEXELS,
        g.texels.empty() ? zero : g.texels.data(),
        g.texels.empty() ? 16 : 4 * g.texels.size(),
        ssbo);
  }
  if (g.nSph) {
    putBuffer(BIND_SPHERES, g.spheres.data(), 16 * size_t(g.nSph), ssbo);
    if (g.flags & F_SPHERE_COLORS)
      putBuffer(BIND_SPHERE_COLORS, g.sphereColors.data(), 12 * size_t(g.nSph), ssbo);
    if (g.flags & F_SPHERE_STYLES)
      putBuffer(BIND_SPHERE_STYLES, g.sphereStyles.data(), 4 * size_t(g.nSph), ssbo);
  }
  if (g.nCyl) {
    putBuffer(BIND_CYLS, g.cyls.data(), 32 * size_t(g.nCyl), ssbo);
    if (g.flags & F_CYL_COLORS)
      putBuffer(BIND_CYL_COLORS, g.cylColors.data(), 24 * size_t(g.nCyl), ssbo);
    if (g.flags & F_CYL_STYLES)
      putBuffer(BIND_CYL_STYLES, g.cylStyles.data(), 4 * size_t(g.nCyl), ssbo);
  }
  if (g.nTri) {
    putBuffer(BIND_TRI_POS,
        g.triPos.data(),
        36 * size_t(g.nTri),
        ssbo | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    putBuffer(BIND_TRI_SHADE, g.triShade.data(), 48 * size_t(g.nTri), ssbo);
    if (!g.triAux.empty())
      putBuffer(BIND_TRI_AUX, g.triAux.data(), 4 * g.triAux.size(), ssbo);
  }
  m_descDirty = true;
}

static VkAccelerationStructureGeometryKHR aabbGeometry(VkDeviceAddress addr)
{
  VkAccelerationStructureGeometryKHR g{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  g.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
  g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  g.geometry.aabbs.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
  g.geometry.aabbs.data.deviceAddress = addr;
  g.geometry.aabbs.stride = 24;
  return g;
}

static VkAccelerationStructureGeometryKHR triangleGeometry(
    VkDeviceAddress addr, uint32_t count)
{
  VkAccelerationStructureGeometryKHR g{
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
  g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
  g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
  g.geometry.triangles.sType =
      VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
  g.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
  g.geometry.triangles.vertexData.deviceAddress = addr;
  g.geometry.triangles.vertexStride = 12;
  g.geometry.triangles.maxVertex = 3 * count - 1;
  g.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
  return g;
}

// Builds (or with 'refitOnly' updates) the BLAS of every slice
void VulkanScene::buildSlices(const Gathered &g, bool updatable, bool refitOnly)
{
  const VkBufferUsageFlags aabbUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
      | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
  if (!refitOnly)
    m_blas.assign(g.slices.size(), Blas());
  for (int type = 0; type < 2; type++) {
    const uint32_t total = type == 0 ? g.nSph : g.nCyl;
    if (!total)
      continue;
    GpuBuffer aabbs = m_ctx->createBuffer(24 * size_t(total), aabbUsage, MemoryKind::DeviceLocal);
    try {
      generateAabbs(uint32_t(type), total, aabbs);
      for (size_t i = 0; i < g.slices.size(); i++) {
        const Slice &s = g.slices[i];
        if (s.type != type)
          continue;
        const auto geom = aabbGeometry(aabbs.address + 24 * VkDeviceAddress(s.base));
        if (refitOnly)
          updateBlas(m_blas[i], geom, s.count);
        else
          m_blas[i] = buildBlas(geom, s.count, updatable);
      }
    } catch (...) {
      m_ctx->destroyBuffer(aabbs);
      throw;
    }
    m_ctx->destroyBuffer(aabbs);
  }
  for (size_t i = 0; i < g.slices.size(); i++) {
    const Slice &s = g.slices[i];
    if (s.type != 2)
      continue;
    const auto geom = triangleGeometry(
        m_buf[BIND_TRI_POS].address + 36 * VkDeviceAddress(s.base), s.count);
    if (refitOnly)
      updateBlas(m_blas[i], geom, s.count);
    else
      m_blas[i] = buildBlas(geom, s.count, updatable);
  }
}

void VulkanScene::ensureLightBuffer()
{
  const size_t need = std::max<size_t>(m_lights.size(), 1);
  if (m_lightBuf && m_lightCapacity >= need)
    return;
  m_ctx->destroyBuffer(m_lightBuf);
  m_lightCapacity = need;
  // rewritten every frame: the unlit headlight follows the camera
  m_lightBuf = m_ctx->createBuffer(
      32 * need, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::Upload);
  m_descDirty = true;
}

void VulkanScene::buildFull(Gathered &g, bool updatable)
{
  const double t0 = nowMs();
  m_ctx->resetPeak();
  destroyScene();
  try {
    uploadSceneData(g);
    const double t1 = nowMs();
    buildSlices(g, updatable, false);
    buildTlas(g);
    m_updatable = updatable;
    m_sceneFlags = g.flags;
    m_styleUniform[0] = g.styleUniform[0];
    m_styleUniform[1] = g.styleUniform[1];
    m_sig = g.sig;
    m_haveSig = true;
    m_refits = 0;
    m_lastFrameHash = 0;
    m_sameFrames = 0;
    if (m_debug) {
      std::fprintf(stderr,
          "[tachyon vulkan] build: %u spheres, %u cylinders, %u triangles in "
          "%zu BLAS, %zu TLAS instances, %zu styles, %zu samplers, %zu "
          "lights: upload %.1f ms, acceleration structures %.1f ms%s\n",
          g.nSph,
          g.nCyl,
          g.nTri,
          g.slices.size(),
          g.instances.size(),
          m_styles.size(),
          m_samplers.size(),
          m_lights.size(),
          t1 - t0,
          nowMs() - t1,
          updatable ? " (updatable)" : "");
      std::fprintf(stderr,
          "[tachyon vulkan] GPU memory resident %.1f MB, peak while building "
          "%.1f MB\n",
          double(m_ctx->deviceBytes()) / 1048576.0,
          double(m_ctx->peakDeviceBytes()) / 1048576.0);
    }
  } catch (...) {
    vkDeviceWaitIdle(m_ctx->device());
    destroyScene();
    throw;
  }
}

void VulkanScene::refit(Gathered &g)
{
  const double t0 = nowMs();
  uploadSceneData(g);
  const double t1 = nowMs();
  buildSlices(g, true, true);
  buildTlas(g);
  m_refits++;
  m_lastFrameHash = 0;
  m_sameFrames = 0;
  if (m_debug)
    std::fprintf(stderr,
        "[tachyon vulkan] refit %u: upload %.1f ms, BLAS update + TLAS %.1f ms\n",
        m_refits,
        t1 - t0,
        nowMs() - t1);
}

void VulkanScene::finishBuild()
{
  Gathered g = gather();
  m_placements.clear();
  if (g.nSph + g.nCyl + g.nTri == 0) {
    destroyScene();
    return;
  }
  const bool same = m_tlas && m_haveSig && g.sig == m_sig;
  try {
    if (same && m_updatable && m_refits < m_maxRefits) {
      refit(g);
    } else {
      // The second build with the same topology in a row: the scene is
      // being animated, so make the structures refittable from now on. Also
      // after the maximum number of refits (refitting degrades the quality).
      const bool animate = same || (m_haveSig && m_updatable);
      buildFull(g, animate);
    }
  } catch (...) {
    vkDeviceWaitIdle(m_ctx->device());
    destroyScene();
    throw;
  }
  ensureLightBuffer();
}

// Frame ///////////////////////////////////////////////////////////////////////

void VulkanScene::setShading(float ambient, float diffuseScale)
{
  m_ambient = ambient;
  m_diffuse = diffuseScale;
}

void VulkanScene::setBackground(const float *rgb)
{
  std::memcpy(m_bg, rgb, 3 * sizeof(float));
}

void VulkanScene::setSamples(int pixelSamples)
{
  m_samples = std::max(1, pixelSamples);
}

void VulkanScene::setCamera(const SceneCamera &cam)
{
  m_cam = cam;
}

void VulkanScene::setOutput(int width, int height, float *)
{
  m_width = width;
  m_height = height;
}

void VulkanScene::setDepthOutput(float *) {}

bool VulkanScene::setDirectOutput(
    OutputFormat format, void *color, float *depth, float bgAlpha)
{
  m_format = color ? format : OutputFormat::None;
  m_colorDst = color;
  m_depthDst = depth;
  m_bg[3] = bgAlpha;
  return true;
}

void VulkanScene::fillDirectBackground()
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

void VulkanScene::render()
{
  if (m_width <= 0 || m_height <= 0)
    return;
  if (!m_tlas) { // nothing renderable
    fillDirectBackground();
    return;
  }
  const double t0 = nowMs();
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

  FrameData fd{};
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
  std::vector<float> lightData;
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
  if (m_accumEnabled && h == m_lastFrameHash)
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
  if (m_sameFrames == 1 && m_samples == 1 && m_debug)
    std::fprintf(stderr, "[tachyon vulkan] accumulating identical frames\n");

  // Resources -----------------------------------------------------------------
  const bool needAccum = accumFlags != 0;
  if (m_outW != m_width || m_outH != m_height || m_outDepth != wantDepth
      || m_outFormat != fmt || (needAccum && !m_accumBuf)) {
    vkDeviceWaitIdle(m_ctx->device());
    const bool keepAccum = needAccum && m_accumBuf && m_outW == m_width
        && m_outH == m_height;
    GpuBuffer keep = m_accumBuf;
    m_accumBuf = GpuBuffer();
    destroyOutputs();
    if (keepAccum)
      m_accumBuf = keep;
    else
      m_ctx->destroyBuffer(keep);
    m_outBuf = m_ctx->createBuffer(pixBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        MemoryKind::DeviceLocal);
    m_readBuf = m_ctx->createBuffer(
        pixBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Readback);
    if (wantDepth) {
      m_depthBuf = m_ctx->createBuffer(4 * np,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          MemoryKind::DeviceLocal);
      m_depthReadBuf = m_ctx->createBuffer(
          4 * np, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Readback);
    }
    if (needAccum && !m_accumBuf)
      m_accumBuf = m_ctx->createBuffer(
          12 * np, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryKind::DeviceLocal);
    m_outW = m_width;
    m_outH = m_height;
    m_outDepth = wantDepth;
    m_outFormat = fmt;
    m_outBytes = pixBytes;
    m_descDirty = true;
    if (m_debug)
      std::fprintf(stderr,
          "[tachyon vulkan] output buffers %.1f MB, resident GPU memory now "
          "%.1f MB\n",
          double(m_outBuf.size + m_depthBuf.size + m_accumBuf.size) / 1048576.0,
          double(m_ctx->deviceBytes()) / 1048576.0);
  }
  if (m_descDirty)
    updateDescriptors();

  std::memcpy(m_frameUbo.mapped, &fd, sizeof(fd));
  m_ctx->syncMapped(m_frameUbo, true);
  if (m_lightBuf.mapped) {
    std::memcpy(m_lightBuf.mapped,
        lightData.data(),
        std::min(lightData.size() * 4, size_t(32) * m_lightCapacity));
    m_ctx->syncMapped(m_lightBuf, true);
  }

  m_ctx->submitAndWait([&](VkCommandBuffer cb) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_ctx->tracePipeline());
    vkCmdBindDescriptorSets(cb,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        m_ctx->pipelineLayout(),
        0,
        1,
        &m_set,
        0,
        nullptr);
    vkCmdDispatch(cb, uint32_t((m_width + 7) / 8), uint32_t((m_height + 7) / 8), 1);
    memoryBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT);
    if (fmt != OutputFormat::None) {
      VkBufferCopy rc{0, 0, pixBytes};
      vkCmdCopyBuffer(cb, m_outBuf.buffer, m_readBuf.buffer, 1, &rc);
    }
    if (wantDepth) {
      VkBufferCopy dc{0, 0, 4 * np};
      vkCmdCopyBuffer(cb, m_depthBuf.buffer, m_depthReadBuf.buffer, 1, &dc);
    }
    memoryBarrier(cb,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_HOST_READ_BIT);
  });
  const double t1 = nowMs();

  if (fmt != OutputFormat::None) {
    m_ctx->syncMapped(m_readBuf, false);
    std::memcpy(m_colorDst, m_readBuf.mapped, pixBytes);
  }
  if (wantDepth) {
    m_ctx->syncMapped(m_depthReadBuf, false);
    std::memcpy(m_depthDst, m_depthReadBuf.mapped, 4 * np);
  }
  if (m_debug)
    std::fprintf(stderr,
        "[tachyon vulkan] frame %dx%d: trace+copy %.2f ms, readback %.2f ms"
        " (samples %u+%u)\n",
        m_width,
        m_height,
        t1 - t0,
        nowMs() - t1,
        base,
        count);
}

} // namespace gpu
} // namespace tachyon_anari
