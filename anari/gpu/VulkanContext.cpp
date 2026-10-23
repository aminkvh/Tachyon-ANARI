// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "VulkanContext.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <vector>

namespace tachyon_anari {
namespace gpu {

namespace {
std::mutex g_mutex;
std::weak_ptr<VulkanContext> g_context;
bool g_failed{false};
bool g_failedHard{false};
std::string g_failReason;

bool hasExtension(
    const std::vector<VkExtensionProperties> &exts, const char *name)
{
  for (const auto &e : exts)
    if (std::strcmp(e.extensionName, name) == 0)
      return true;
  return false;
}
} // namespace

std::shared_ptr<VulkanContext> VulkanContext::acquire(
    std::string &reason, bool &hard)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (auto ctx = g_context.lock())
    return ctx;
  if (g_failed) { // do not probe the driver again and again
    reason = g_failReason;
    hard = g_failedHard;
    return nullptr;
  }
  std::shared_ptr<VulkanContext> ctx(new VulkanContext);
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = ctx->init(reason, hard);
  if (const char *dbg = std::getenv("TACHYON_VULKAN_DEBUG"))
    if (*dbg && *dbg != '0')
      std::fprintf(stderr, "[tachyon vulkan] context init %.0f ms\n",
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - t0)
              .count());
  if (!ok) {
    g_failed = true;
    g_failReason = reason;
    g_failedHard = hard;
    return nullptr;
  }
  g_context = ctx;
  return ctx;
}

bool VulkanContext::init(std::string &reason, bool &hard)
{
  hard = false;
  if (volkInitialize() != VK_SUCCESS) {
    reason = "no Vulkan loader found";
    return false;
  }
  if (volkGetInstanceVersion() < VK_API_VERSION_1_2) {
    reason = "Vulkan 1.2 or newer is required";
    return false;
  }

  VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  app.pApplicationName = "tachyon-anari";
  app.apiVersion = VK_API_VERSION_1_3;
  VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  ici.pApplicationInfo = &app;
  VkResult res = vkCreateInstance(&ici, nullptr, &m_instance);
  if (res != VK_SUCCESS) {
    reason = "vkCreateInstance failed (" + std::to_string(int(res)) + ")";
    hard = true;
    return false;
  }
  volkLoadInstanceOnly(m_instance);

  uint32_t count = 0;
  vkEnumeratePhysicalDevices(m_instance, &count, nullptr);
  std::vector<VkPhysicalDevice> devices(count);
  vkEnumeratePhysicalDevices(m_instance, &count, devices.data());
  if (devices.empty()) {
    reason = "no Vulkan devices";
    return false;
  }

  // TACHYON_VULKAN_DEVICE=<index>: only consider that device (diagnostics)
  int forced = -1;
  if (const char *env = std::getenv("TACHYON_VULKAN_DEVICE"))
    if (*env)
      forced = std::atoi(env);

  std::string why;
  for (uint32_t i = 0; i < devices.size(); i++) {
    if (forced >= 0 && int(i) != forced)
      continue;
    VkPhysicalDevice pd = devices[i];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(pd, &props);
    const std::string name = props.deviceName;
    if (props.apiVersion < VK_API_VERSION_1_2) {
      why += name + ": Vulkan 1.2 missing; ";
      continue;
    }
    uint32_t en = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, nullptr);
    std::vector<VkExtensionProperties> exts(en);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &en, exts.data());
    if (!hasExtension(exts, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
        || !hasExtension(exts, VK_KHR_RAY_QUERY_EXTENSION_NAME)
        || !hasExtension(
            exts, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)) {
      why += name + ": no ray query / acceleration structure support; ";
      continue;
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rq{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features f12{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    rq.pNext = &as;
    as.pNext = &f12;
    f2.pNext = &rq;
    vkGetPhysicalDeviceFeatures2(pd, &f2);
    if (!rq.rayQuery || !as.accelerationStructure
        || !f12.bufferDeviceAddress) {
      why += name + ": required ray tracing features missing; ";
      continue;
    }

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf.data());
    uint32_t family = ~0u;
    for (uint32_t q = 0; q < nq; q++)
      if (qf[q].queueFlags & VK_QUEUE_COMPUTE_BIT) {
        family = q;
        break;
      }
    if (family == ~0u) {
      why += name + ": no compute queue; ";
      continue;
    }

    // Enable exactly what the backend uses
    VkPhysicalDeviceRayQueryFeaturesKHR rqOn{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rqOn.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asOn{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asOn.accelerationStructure = VK_TRUE;
    asOn.pNext = &rqOn;
    VkPhysicalDeviceVulkan12Features f12On{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12On.bufferDeviceAddress = VK_TRUE;
    f12On.pNext = &asOn;

    const float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    const char *devExts[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f12On;
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 3;
    dci.ppEnabledExtensionNames = devExts;
    res = vkCreateDevice(pd, &dci, nullptr, &m_device);
    if (res != VK_SUCCESS) {
      why += name + ": vkCreateDevice failed; ";
      continue;
    }

    m_phys = pd;
    m_family = family;
    m_gpuName = name;
    volkLoadDevice(m_device);
    vkGetDeviceQueue(m_device, family, 0, &m_queue);
    vkGetPhysicalDeviceMemoryProperties(pd, &m_mem);

    VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 p2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &asp;
    vkGetPhysicalDeviceProperties2(pd, &p2);
    m_scratchAlign =
        std::max(1u, asp.minAccelerationStructureScratchOffsetAlignment);

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = family;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateCommandPool(m_device, &pci, nullptr, &m_pool) != VK_SUCCESS
        || vkCreateFence(m_device, &fci, nullptr, &m_fence) != VK_SUCCESS) {
      reason = "could not create Vulkan command pool";
      hard = true;
      return false;
    }
    return true;
  }

  if (forced >= 0 && why.empty())
    why = "TACHYON_VULKAN_DEVICE=" + std::to_string(forced) + " does not exist";
  reason = "no Vulkan device with ray tracing support (" + why + ")";
  return false;
}

VulkanContext::~VulkanContext()
{
  if (m_device) {
    vkDeviceWaitIdle(m_device);
    if (m_fence)
      vkDestroyFence(m_device, m_fence, nullptr);
    if (m_pool)
      vkDestroyCommandPool(m_device, m_pool, nullptr);
    vkDestroyDevice(m_device, nullptr);
  }
  if (m_instance)
    vkDestroyInstance(m_instance, nullptr);
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits,
    VkMemoryPropertyFlags want,
    VkMemoryPropertyFlags avoid) const
{
  for (uint32_t i = 0; i < m_mem.memoryTypeCount; i++) {
    const VkMemoryPropertyFlags f = m_mem.memoryTypes[i].propertyFlags;
    if ((typeBits & (1u << i)) && (f & want) == want && !(f & avoid))
      return i;
  }
  return ~0u;
}

GpuBuffer VulkanContext::createBuffer(
    VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind)
{
  GpuBuffer b;
  VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  bci.size = std::max<VkDeviceSize>(size, 16);
  bci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  if (vkCreateBuffer(m_device, &bci, nullptr, &b.buffer) != VK_SUCCESS)
    throw std::runtime_error("vkCreateBuffer failed");

  VkMemoryRequirements req;
  vkGetBufferMemoryRequirements(m_device, b.buffer, &req);

  uint32_t type = ~0u;
  const VkMemoryPropertyFlags dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  const VkMemoryPropertyFlags hc = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const VkMemoryPropertyFlags cached = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  switch (kind) {
  case MemoryKind::DeviceLocal:
    type = findMemoryType(req.memoryTypeBits, dl, hv);
    if (type == ~0u)
      type = findMemoryType(req.memoryTypeBits, dl, 0);
    b.deviceLocal = true;
    break;
  case MemoryKind::Upload:
    type = findMemoryType(req.memoryTypeBits, hv | hc, dl | cached);
    if (type == ~0u)
      type = findMemoryType(req.memoryTypeBits, hv | hc, 0);
    break;
  case MemoryKind::Readback:
    type = findMemoryType(req.memoryTypeBits, hv | cached, dl);
    if (type == ~0u)
      type = findMemoryType(req.memoryTypeBits, hv, dl);
    break;
  }
  if (type == ~0u) {
    vkDestroyBuffer(m_device, b.buffer, nullptr);
    throw std::runtime_error("no suitable Vulkan memory type");
  }

  VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
  flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
  VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  mai.pNext = &flags;
  mai.allocationSize = req.size;
  mai.memoryTypeIndex = type;
  if (vkAllocateMemory(m_device, &mai, nullptr, &b.memory) != VK_SUCCESS) {
    vkDestroyBuffer(m_device, b.buffer, nullptr);
    throw std::runtime_error(
        "Vulkan out of memory (" + std::to_string(size >> 20) + " MB)");
  }
  vkBindBufferMemory(m_device, b.buffer, b.memory, 0);

  b.size = req.size; // the accounted size, used again when destroying
  b.hostCoherent = (m_mem.memoryTypes[type].propertyFlags & hc) != 0;
  if (kind != MemoryKind::DeviceLocal) {
    if (vkMapMemory(m_device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped)
        != VK_SUCCESS)
      b.mapped = nullptr;
  } else {
    m_deviceBytes += size_t(req.size);
    m_peakDeviceBytes = std::max(m_peakDeviceBytes, m_deviceBytes);
  }
  VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
  dai.buffer = b.buffer;
  b.address = vkGetBufferDeviceAddress(m_device, &dai);
  return b;
}

void VulkanContext::destroyBuffer(GpuBuffer &b)
{
  if (!b.buffer)
    return;
  if (b.mapped)
    vkUnmapMemory(m_device, b.memory);
  vkDestroyBuffer(m_device, b.buffer, nullptr);
  vkFreeMemory(m_device, b.memory, nullptr);
  if (b.deviceLocal)
    m_deviceBytes -= size_t(b.size);
  b = GpuBuffer();
}

void VulkanContext::syncMapped(GpuBuffer &b, bool toDevice)
{
  if (b.hostCoherent || !b.mapped)
    return;
  VkMappedMemoryRange r{VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE};
  r.memory = b.memory;
  r.offset = 0;
  r.size = VK_WHOLE_SIZE;
  if (toDevice)
    vkFlushMappedMemoryRanges(m_device, 1, &r);
  else
    vkInvalidateMappedMemoryRanges(m_device, 1, &r);
}

VkCommandBuffer VulkanContext::beginCommands()
{
  VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
  ai.commandPool = m_pool;
  ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  ai.commandBufferCount = 1;
  VkCommandBuffer cb;
  if (vkAllocateCommandBuffers(m_device, &ai, &cb) != VK_SUCCESS)
    throw std::runtime_error("vkAllocateCommandBuffers failed");
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cb, &bi);
  return cb;
}

void VulkanContext::endAndWait(VkCommandBuffer cb)
{
  vkEndCommandBuffer(cb);
  VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  si.commandBufferCount = 1;
  si.pCommandBuffers = &cb;
  vkResetFences(m_device, 1, &m_fence);
  VkResult r = vkQueueSubmit(m_queue, 1, &si, m_fence);
  if (r == VK_SUCCESS)
    r = vkWaitForFences(m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);
  vkFreeCommandBuffers(m_device, m_pool, 1, &cb);
  if (r != VK_SUCCESS)
    throw std::runtime_error(
        "Vulkan submit failed (" + std::to_string(int(r)) + ")");
}

} // namespace gpu
} // namespace tachyon_anari
