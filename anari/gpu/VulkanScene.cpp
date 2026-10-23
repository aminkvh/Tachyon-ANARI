// Copyright (c) 2026 Tachyon-ANARI contributors
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

static_assert(NUM_BINDINGS == kNumBindings, "descriptor layout");

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

void VulkanScene::render()
{
  if (m_width <= 0 || m_height <= 0)
    return;
  if (!m_tlas) { // nothing renderable
    fillDirectBackground();
    return;
  }
  const double t0 = nowMs();
  FrameSetup setup = prepareFrame();
  const size_t np = setup.numPixels;
  const bool wantDepth = setup.depth;
  const OutputFormat fmt = setup.format;
  const size_t pixBytes = setup.pixelBytes;
  const FrameData &fd = setup.fd;
  const std::vector<float> &lightData = setup.lights;
  const uint32_t base = setup.base, count = setup.count;

  // Resources -----------------------------------------------------------------
  const bool needAccum = setup.accumulate;
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

