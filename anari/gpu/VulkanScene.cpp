// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "VulkanScene.h"

#include "shaders/raytrace.spv.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace tachyon_anari {
namespace gpu {

namespace {

// Matches PushConstants in shaders/raytrace.slang
struct PushConstants
{
  float posOrtho[4];
  float fwdU0[4];
  float rightDu[4];
  float upV0[4];
  float shade[4];
  float bg[4];
  float baseColor[4];
  uint32_t size[2];
  uint32_t flags[2];
};
static_assert(sizeof(PushConstants) == 128, "push constant block size");

enum Warning
{
  WARN_TRIANGLES = 1,
  WARN_CYLINDERS = 2,
  WARN_LIGHTS = 4,
  WARN_SAMPLES = 8,
  WARN_OPACITY = 16
};

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
  if (r != VK_SUCCESS)
    throw std::runtime_error(
        std::string(what) + " failed (" + std::to_string(int(r)) + ")");
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
  try {
    createPipelines();
  } catch (...) {
    // the destructor cleans up what exists
    throw;
  }
}

VulkanScene::~VulkanScene()
{
  VkDevice dev = m_ctx->device();
  vkDeviceWaitIdle(dev);
  destroyScene();
  destroyOutputs();
  if (m_descPool)
    vkDestroyDescriptorPool(dev, m_descPool, nullptr);
  if (m_aabbPipe)
    vkDestroyPipeline(dev, m_aabbPipe, nullptr);
  if (m_tracePipe)
    vkDestroyPipeline(dev, m_tracePipe, nullptr);
  if (m_pipeLayout)
    vkDestroyPipelineLayout(dev, m_pipeLayout, nullptr);
  if (m_setLayout)
    vkDestroyDescriptorSetLayout(dev, m_setLayout, nullptr);
  if (m_module)
    vkDestroyShaderModule(dev, m_module, nullptr);
}

void VulkanScene::createPipelines()
{
  VkDevice dev = m_ctx->device();

  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = size_t(kRaytraceSpvSize);
  smi.pCode = reinterpret_cast<const uint32_t *>(kRaytraceSpv);
  check(vkCreateShaderModule(dev, &smi, nullptr, &m_module), "vkCreateShaderModule");

  VkDescriptorSetLayoutBinding bind[6]{};
  for (uint32_t i = 0; i < 6; i++) {
    bind[i].binding = i;
    bind[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                                    : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind[i].descriptorCount = 1;
    bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dsl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dsl.bindingCount = 6;
  dsl.pBindings = bind;
  check(vkCreateDescriptorSetLayout(dev, &dsl, nullptr, &m_setLayout),
      "vkCreateDescriptorSetLayout");

  VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants)};
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  pli.pSetLayouts = &m_setLayout;
  pli.pushConstantRangeCount = 1;
  pli.pPushConstantRanges = &pcr;
  check(vkCreatePipelineLayout(dev, &pli, nullptr, &m_pipeLayout),
      "vkCreatePipelineLayout");

  const char *names[2] = {"genAabbs", "main"};
  VkPipeline *pipes[2] = {&m_aabbPipe, &m_tracePipe};
  for (int i = 0; i < 2; i++) {
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m_module;
    ci.stage.pName = names[i];
    ci.layout = m_pipeLayout;
    check(vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, nullptr, pipes[i]),
        "vkCreateComputePipelines");
  }

  VkDescriptorPoolSize sizes[2] = {
      {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5}};
  VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  dpi.maxSets = 1;
  dpi.poolSizeCount = 2;
  dpi.pPoolSizes = sizes;
  check(vkCreateDescriptorPool(dev, &dpi, nullptr, &m_descPool),
      "vkCreateDescriptorPool");
  VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  dai.descriptorPool = m_descPool;
  dai.descriptorSetCount = 1;
  dai.pSetLayouts = &m_setLayout;
  check(vkAllocateDescriptorSets(dev, &dai, &m_set), "vkAllocateDescriptorSets");
}

void VulkanScene::destroyScene()
{
  VkDevice dev = m_ctx->device();
  if (m_tlas)
    vkDestroyAccelerationStructureKHR(dev, m_tlas, nullptr);
  if (m_blas)
    vkDestroyAccelerationStructureKHR(dev, m_blas, nullptr);
  m_tlas = m_blas = VK_NULL_HANDLE;
  m_ctx->destroyBuffer(m_tlasBuf);
  m_ctx->destroyBuffer(m_blasBuf);
  m_ctx->destroyBuffer(m_sphereBuf);
  m_ctx->destroyBuffer(m_colorBuf);
  m_numSpheresBuilt = 0;
  m_descSceneDirty = true;
}

void VulkanScene::destroyOutputs()
{
  m_ctx->destroyBuffer(m_outBuf);
  m_ctx->destroyBuffer(m_depthBuf);
  m_ctx->destroyBuffer(m_readBuf);
  m_ctx->destroyBuffer(m_depthReadBuf);
  m_outW = m_outH = 0;
  m_descOutDirty = true;
}

void VulkanScene::warnOnce(int id, const char *text)
{
  if (m_warned & unsigned(id))
    return;
  m_warned |= unsigned(id);
  m_warnings.emplace_back(text);
}

std::vector<std::string> VulkanScene::takeWarnings()
{
  std::vector<std::string> w;
  w.swap(m_warnings);
  return w;
}

// Scene description ///////////////////////////////////////////////////////////

void VulkanScene::reset(bool lit, int)
{
  vkDeviceWaitIdle(m_ctx->device());
  destroyScene();
  m_styles.clear();
  std::vector<float>().swap(m_spheres);
  std::vector<float>().swap(m_colors);
  m_runs.clear();
  m_colorsActive = false;
  m_numSpheres = m_numTriangles = m_numCylinders = 0;
  m_lit = lit;
  if (lit)
    warnOnce(WARN_LIGHTS,
        "the Vulkan backend does not support lights yet and renders the "
        "eyeLight shading instead");
}

int VulkanScene::addStyle(const SurfaceStyle &style)
{
  m_styles.push_back(style);
  if (style.opacity < 1.f)
    warnOnce(WARN_OPACITY,
        "the Vulkan backend does not support transparency yet, surfaces are "
        "rendered opaque");
  return int(m_styles.size()) - 1;
}

void VulkanScene::addTriangle(int, const float *, const float *, const float *)
{
  m_numTriangles++;
  warnOnce(WARN_TRIANGLES,
      "the Vulkan backend only renders spheres so far: triangles are skipped");
}

void VulkanScene::addSampledTriangle(int,
    const float *,
    const float *,
    const float *,
    AttrSampleFn,
    const void *)
{
  m_numTriangles++;
  warnOnce(WARN_TRIANGLES,
      "the Vulkan backend only renders spheres so far: triangles are skipped");
}

void VulkanScene::addCylinder(int,
    const float *,
    const float *,
    float,
    const float *,
    const float *,
    int)
{
  m_numCylinders++;
  warnOnce(WARN_CYLINDERS,
      "the Vulkan backend only renders spheres so far: cylinders are skipped");
}

void VulkanScene::addDirectionalLight(const float *, const float *)
{
  warnOnce(WARN_LIGHTS,
      "the Vulkan backend does not support lights yet and renders the "
      "eyeLight shading instead");
}

void VulkanScene::addPointLight(const float *, const float *)
{
  warnOnce(WARN_LIGHTS,
      "the Vulkan backend does not support lights yet and renders the "
      "eyeLight shading instead");
}

// Colors are only stored per sphere when they are not all the same.
void VulkanScene::ensureColors()
{
  if (m_colorsActive)
    return;
  m_colorsActive = true;
  m_colors.reserve(m_spheres.capacity() / 4 * 3);
  for (const auto &run : m_runs) {
    const float *c = m_styles[run.second].color;
    for (size_t i = 0; i < run.first; i++)
      m_colors.insert(m_colors.end(), c, c + 3);
  }
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
  const float *styleColor = m_styles[style].color;
  if (m_spheres.empty())
    std::memcpy(m_baseColor, styleColor, sizeof(m_baseColor));
  else if (!m_colorsActive && !rgb
      && std::memcmp(m_baseColor, styleColor, sizeof(m_baseColor)) != 0)
    ensureColors(); // spheres of different styles differ in color
  if (rgb)
    ensureColors();

  m_spheres.reserve(m_spheres.size() + 4 * count);
  if (m_colorsActive)
    m_colors.reserve(m_colors.size() + 3 * count);
  size_t kept = 0;
  for (size_t i = 0; i < count; i++) {
    const float r = radius ? radius[i] : uniformRadius;
    if (!(r > 0.f))
      continue;
    m_spheres.push_back(center[3 * i]);
    m_spheres.push_back(center[3 * i + 1]);
    m_spheres.push_back(center[3 * i + 2]);
    m_spheres.push_back(r);
    if (m_colorsActive) {
      const float *c = rgb ? rgb + 3 * i : styleColor;
      m_colors.insert(m_colors.end(), c, c + 3);
    }
    kept++;
  }
  m_runs.emplace_back(kept, style);
  m_numSpheres += kept;
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

void VulkanScene::finishBuild()
{
  if (m_spheres.empty())
    return;
  const double t0 = nowMs();
  VkDevice dev = m_ctx->device();
  const size_t n = m_spheres.size() / 4;
  m_numSpheresBuilt = n;
  m_ctx->resetPeak();

  m_sphereBuf = m_ctx->createBuffer(16 * n,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      MemoryKind::DeviceLocal);
  upload(m_sphereBuf, m_spheres.data(), 16 * n);
  std::vector<float>().swap(m_spheres);
  if (m_colorsActive) {
    m_colorBuf = m_ctx->createBuffer(12 * n,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        MemoryKind::DeviceLocal);
    upload(m_colorBuf, m_colors.data(), 12 * n);
    std::vector<float>().swap(m_colors);
  }
  const double t1 = nowMs();

  // AABBs for all spheres, generated on the GPU
  GpuBuffer aabbs = m_ctx->createBuffer(24 * n,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
          | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
      MemoryKind::DeviceLocal);
  GpuBuffer scratch, instances;
  VkQueryPool queries = VK_NULL_HANDLE;
  GpuBuffer oldBlasBuf;
  VkAccelerationStructureKHR oldBlas = VK_NULL_HANDLE;
  VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
  GpuBuffer blasBuf;
  try {
    {
      VkDescriptorBufferInfo bi[2] = {{m_sphereBuf.buffer, 0, VK_WHOLE_SIZE},
          {aabbs.buffer, 0, VK_WHOLE_SIZE}};
      VkWriteDescriptorSet w[2]{};
      for (int i = 0; i < 2; i++) {
        w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = m_set;
        w[i].dstBinding = i == 0 ? 1 : 5;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
      }
      vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
      PushConstants pc{};
      pc.size[0] = uint32_t(n);
      m_ctx->submitAndWait([&](VkCommandBuffer cb) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_aabbPipe);
        vkCmdBindDescriptorSets(cb,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            m_pipeLayout,
            0,
            1,
            &m_set,
            0,
            nullptr);
        vkCmdPushConstants(
            cb, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cb, uint32_t((n + 63) / 64), 1, 1);
        memoryBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_ACCESS_SHADER_WRITE_BIT,
            VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
            VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                | VK_ACCESS_SHADER_READ_BIT);
      });
    }
    const double t2 = nowMs();

    // BLAS
    VkAccelerationStructureGeometryKHR geom{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geom.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    geom.geometry.aabbs.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geom.geometry.aabbs.data.deviceAddress = aabbs.address;
    geom.geometry.aabbs.stride = 24;
    VkAccelerationStructureBuildGeometryInfoKHR bgi{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bgi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    bgi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
    bgi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bgi.geometryCount = 1;
    bgi.pGeometries = &geom;
    const uint32_t primCount = uint32_t(n);
    VkAccelerationStructureBuildSizesInfoKHR sz{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(dev,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &bgi,
        &primCount,
        &sz);

    const VkDeviceSize align = m_ctx->scratchAlignment();
    scratch = m_ctx->createBuffer(sz.buildScratchSize + align,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        MemoryKind::DeviceLocal);
    bgi.scratchData.deviceAddress = (scratch.address + align - 1) / align * align;

    oldBlasBuf = m_ctx->createBuffer(sz.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        MemoryKind::DeviceLocal);
    VkAccelerationStructureCreateInfoKHR asci{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    asci.buffer = oldBlasBuf.buffer;
    asci.size = sz.accelerationStructureSize;
    asci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    check(vkCreateAccelerationStructureKHR(dev, &asci, nullptr, &oldBlas),
        "vkCreateAccelerationStructureKHR");
    bgi.dstAccelerationStructure = oldBlas;

    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
    qpi.queryCount = 1;
    check(vkCreateQueryPool(dev, &qpi, nullptr, &queries), "vkCreateQueryPool");

    VkAccelerationStructureBuildRangeInfoKHR range{primCount, 0, 0, 0};
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
          &oldBlas,
          VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
          queries,
          0);
    });
    const double t3 = nowMs();

    // compaction
    VkDeviceSize compactSize = 0;
    check(vkGetQueryPoolResults(dev,
              queries,
              0,
              1,
              sizeof(compactSize),
              &compactSize,
              sizeof(compactSize),
              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
        "vkGetQueryPoolResults");
    if (compactSize == 0 || compactSize > sz.accelerationStructureSize)
      compactSize = sz.accelerationStructureSize;
    blasBuf = m_ctx->createBuffer(compactSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
        MemoryKind::DeviceLocal);
    asci.buffer = blasBuf.buffer;
    asci.size = compactSize;
    check(vkCreateAccelerationStructureKHR(dev, &asci, nullptr, &blas),
        "vkCreateAccelerationStructureKHR");
    m_ctx->submitAndWait([&](VkCommandBuffer cb) {
      VkCopyAccelerationStructureInfoKHR ci{
          VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR};
      ci.src = oldBlas;
      ci.dst = blas;
      ci.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;
      vkCmdCopyAccelerationStructureKHR(cb, &ci);
    });
    const double t4 = nowMs();

    vkDestroyAccelerationStructureKHR(dev, oldBlas, nullptr);
    oldBlas = VK_NULL_HANDLE;
    m_ctx->destroyBuffer(oldBlasBuf);
    m_ctx->destroyBuffer(scratch);
    m_ctx->destroyBuffer(aabbs);
    vkDestroyQueryPool(dev, queries, nullptr);
    queries = VK_NULL_HANDLE;

    // TLAS with a single identity instance of the BLAS
    VkAccelerationStructureDeviceAddressInfoKHR dai{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    dai.accelerationStructure = blas;
    VkAccelerationStructureInstanceKHR inst{};
    inst.transform.matrix[0][0] = inst.transform.matrix[1][1] =
        inst.transform.matrix[2][2] = 1.f;
    inst.mask = 0xFF;
    inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
    inst.accelerationStructureReference =
        vkGetAccelerationStructureDeviceAddressKHR(dev, &dai);
    instances = m_ctx->createBuffer(sizeof(inst),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
        MemoryKind::Upload);
    std::memcpy(instances.mapped, &inst, sizeof(inst));
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
    const uint32_t one = 1;
    VkAccelerationStructureBuildSizesInfoKHR tsz{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    vkGetAccelerationStructureBuildSizesKHR(dev,
        VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &tbi,
        &one,
        &tsz);
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
    VkAccelerationStructureBuildRangeInfoKHR trange{1, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR *pTRange = &trange;
    m_ctx->submitAndWait([&](VkCommandBuffer cb) {
      vkCmdBuildAccelerationStructuresKHR(cb, 1, &tbi, &pTRange);
      memoryBarrier(cb,
          VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
          VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    });
    m_ctx->destroyBuffer(scratch);
    m_ctx->destroyBuffer(instances);

    m_blas = blas;
    m_blasBuf = blasBuf;
    m_descSceneDirty = true;
    const double t5 = nowMs();

    if (m_debug) {
      std::fprintf(stderr,
          "[tachyon vulkan] %zu spheres: upload %.1f ms, aabbs %.1f ms, "
          "BLAS build %.1f ms, compaction %.1f ms (%.1f -> %.1f MB), TLAS "
          "%.1f ms, total %.1f ms\n",
          n,
          t1 - t0,
          t2 - t1,
          t3 - t2,
          t4 - t3,
          double(sz.accelerationStructureSize) / 1048576.0,
          double(compactSize) / 1048576.0,
          t5 - t4,
          t5 - t0);
      std::fprintf(stderr,
          "[tachyon vulkan] GPU memory: spheres %.1f MB, colors %.1f MB, "
          "BLAS %.1f MB, TLAS %.2f MB; resident %.1f MB, peak while "
          "building %.1f MB\n",
          double(m_sphereBuf.size) / 1048576.0,
          double(m_colorBuf.size) / 1048576.0,
          double(m_blasBuf.size) / 1048576.0,
          double(m_tlasBuf.size) / 1048576.0,
          double(m_ctx->deviceBytes()) / 1048576.0,
          double(m_ctx->peakDeviceBytes()) / 1048576.0);
    }
  } catch (...) {
    vkDeviceWaitIdle(dev);
    if (oldBlas)
      vkDestroyAccelerationStructureKHR(dev, oldBlas, nullptr);
    if (blas)
      vkDestroyAccelerationStructureKHR(dev, blas, nullptr);
    if (queries)
      vkDestroyQueryPool(dev, queries, nullptr);
    m_ctx->destroyBuffer(oldBlasBuf);
    m_ctx->destroyBuffer(blasBuf);
    m_ctx->destroyBuffer(scratch);
    m_ctx->destroyBuffer(instances);
    m_ctx->destroyBuffer(aabbs);
    destroyScene();
    throw;
  }
}

// Frame ///////////////////////////////////////////////////////////////////////

void VulkanScene::setShading(float ambient, float diffuseScale)
{
  m_ambient = ambient;
  m_diffuse = diffuseScale;
}

void VulkanScene::setBackground(const float *rgb)
{
  std::memcpy(m_bg, rgb, sizeof(m_bg));
}

void VulkanScene::setSamples(int pixelSamples)
{
  if (pixelSamples > 1)
    warnOnce(WARN_SAMPLES,
        "the Vulkan backend renders one sample per pixel: 'pixelSamples' is "
        "ignored");
}

void VulkanScene::setCamera(const SceneCamera &cam)
{
  m_cam = cam;
}

void VulkanScene::setOutput(int width, int height, float *rgbBuffer)
{
  m_width = width;
  m_height = height;
  m_rgb = rgbBuffer;
}

void VulkanScene::setDepthOutput(float *depthBuffer)
{
  m_depth = depthBuffer;
}

void VulkanScene::fillBackground()
{
  const size_t np = size_t(m_width) * size_t(m_height);
  for (size_t i = 0; i < np; i++) {
    m_rgb[3 * i] = m_bg[0];
    m_rgb[3 * i + 1] = m_bg[1];
    m_rgb[3 * i + 2] = m_bg[2];
  }
  if (m_depth)
    std::fill(m_depth, m_depth + np, 1e18f);
}

void VulkanScene::updateDescriptors()
{
  VkDevice dev = m_ctx->device();
  VkWriteDescriptorSetAccelerationStructureKHR asw{
      VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
  asw.accelerationStructureCount = 1;
  asw.pAccelerationStructures = &m_tlas;
  // unused buffers (no colors, no depth) point at a buffer that exists
  VkDescriptorBufferInfo bi[5] = {
      {m_sphereBuf.buffer, 0, VK_WHOLE_SIZE},
      {m_colorBuf ? m_colorBuf.buffer : m_sphereBuf.buffer, 0, VK_WHOLE_SIZE},
      {m_outBuf.buffer, 0, VK_WHOLE_SIZE},
      {m_outDepth ? m_depthBuf.buffer : m_outBuf.buffer, 0, VK_WHOLE_SIZE},
      {m_sphereBuf.buffer, 0, VK_WHOLE_SIZE}};
  VkWriteDescriptorSet w[6]{};
  w[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  w[0].pNext = &asw;
  w[0].dstSet = m_set;
  w[0].dstBinding = 0;
  w[0].descriptorCount = 1;
  w[0].descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
  for (int i = 0; i < 5; i++) {
    w[i + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w[i + 1].dstSet = m_set;
    w[i + 1].dstBinding = uint32_t(i + 1);
    w[i + 1].descriptorCount = 1;
    w[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[i + 1].pBufferInfo = &bi[i];
  }
  vkUpdateDescriptorSets(dev, 6, w, 0, nullptr);
  m_descSceneDirty = m_descOutDirty = false;
}

void VulkanScene::render()
{
  if (m_width <= 0 || m_height <= 0 || !m_rgb)
    return;
  if (!m_tlas) { // nothing renderable
    fillBackground();
    return;
  }
  const double t0 = nowMs();
  const size_t np = size_t(m_width) * size_t(m_height);
  const bool wantDepth = m_depth != nullptr;

  if (m_outW != m_width || m_outH != m_height || m_outDepth != wantDepth) {
    vkDeviceWaitIdle(m_ctx->device());
    destroyOutputs();
    m_outBuf = m_ctx->createBuffer(12 * np,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        MemoryKind::DeviceLocal);
    m_readBuf = m_ctx->createBuffer(
        12 * np, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Readback);
    if (wantDepth) {
      m_depthBuf = m_ctx->createBuffer(4 * np,
          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
          MemoryKind::DeviceLocal);
      m_depthReadBuf = m_ctx->createBuffer(
          4 * np, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryKind::Readback);
    }
    m_outW = m_width;
    m_outH = m_height;
    m_outDepth = wantDepth;
    m_descOutDirty = true;
    if (m_debug)
      std::fprintf(stderr,
          "[tachyon vulkan] output buffers %.1f MB, resident GPU memory now "
          "%.1f MB\n",
          double(m_outBuf.size + m_depthBuf.size) / 1048576.0,
          double(m_ctx->deviceBytes()) / 1048576.0);
  }
  if (m_descSceneDirty || m_descOutDirty)
    updateDescriptors();

  // Camera: pixel (x, y from the top) looks through the image plane point
  // u * right + v * up at distance 1 (perspective) or from there (orthographic)
  const SceneCamera &c = m_cam;
  auto norm = [](const float *a, float *o) {
    const float l = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    const float s = l > 0.f ? 1.f / l : 0.f;
    o[0] = a[0] * s;
    o[1] = a[1] * s;
    o[2] = a[2] * s;
  };
  float f[3], r[3], u[3], t[3];
  norm(c.direction, f);
  const float rr[3] = {f[1] * c.up[2] - f[2] * c.up[1],
      f[2] * c.up[0] - f[0] * c.up[2],
      f[0] * c.up[1] - f[1] * c.up[0]};
  norm(rr, r);
  t[0] = r[1] * f[2] - r[2] * f[1];
  t[1] = r[2] * f[0] - r[0] * f[2];
  t[2] = r[0] * f[1] - r[1] * f[0];
  norm(t, u);
  const float frameAspect = float(m_width) / float(m_height);
  const float aspect = c.aspect > 0.f ? c.aspect : frameAspect;
  const float ph = c.orthographic ? (c.height > 0.f ? c.height : 1.f)
                                  : 2.f * std::tan(0.5f * c.fovy);
  const float pw = aspect * ph;
  const float *reg = c.region;

  PushConstants pc{};
  for (int i = 0; i < 3; i++) {
    pc.posOrtho[i] = c.position[i];
    pc.fwdU0[i] = f[i];
    pc.rightDu[i] = r[i];
    pc.upV0[i] = u[i];
    pc.bg[i] = m_bg[i];
    pc.baseColor[i] = m_baseColor[i];
  }
  pc.posOrtho[3] = c.orthographic ? 1.f : 0.f;
  pc.fwdU0[3] = (reg[0] - 0.5f) * pw;
  pc.rightDu[3] = (reg[2] - reg[0]) * pw / float(m_width);
  pc.upV0[3] = (reg[3] - 0.5f) * ph;
  pc.shade[0] = m_ambient;
  pc.shade[1] = m_diffuse;
  pc.shade[2] = -(reg[3] - reg[1]) * ph / float(m_height);
  // The CPU backend (verified against helide with the CTS) shoots each ray
  // through the lower-left corner of its pixel in ANARI's orientation, i.e.
  // half a pixel left of and half a pixel below the pixel center the shader
  // uses. Measured: this reproduces the CPU image to 91 dB.
  pc.fwdU0[3] -= 0.5f * pc.rightDu[3];
  pc.upV0[3] += 0.5f * pc.shade[2]; // shade[2] (v step) is negative: down
  pc.size[0] = uint32_t(m_width);
  pc.size[1] = uint32_t(m_height);
  pc.flags[0] = m_colorBuf ? 1u : 0u;
  pc.flags[1] = wantDepth ? 1u : 0u;

  m_ctx->submitAndWait([&](VkCommandBuffer cb) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipe);
    vkCmdBindDescriptorSets(cb,
        VK_PIPELINE_BIND_POINT_COMPUTE,
        m_pipeLayout,
        0,
        1,
        &m_set,
        0,
        nullptr);
    vkCmdPushConstants(
        cb, m_pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cb, uint32_t((m_width + 7) / 8), uint32_t((m_height + 7) / 8), 1);
    memoryBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferCopy rc{0, 0, 12 * np};
    vkCmdCopyBuffer(cb, m_outBuf.buffer, m_readBuf.buffer, 1, &rc);
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

  m_ctx->syncMapped(m_readBuf, false);
  std::memcpy(m_rgb, m_readBuf.mapped, 12 * np);
  if (wantDepth) {
    m_ctx->syncMapped(m_depthReadBuf, false);
    std::memcpy(m_depth, m_depthReadBuf.mapped, 4 * np);
  }
  if (m_debug)
    std::fprintf(stderr,
        "[tachyon vulkan] frame %dx%d: trace+copy %.2f ms, readback %.2f ms\n",
        m_width,
        m_height,
        t1 - t0,
        nowMs() - t1);
}

} // namespace gpu
} // namespace tachyon_anari
