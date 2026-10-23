// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#include "VulkanContext.h"

#include "../Backends.h"
#include "shaders/raytrace.spv.h"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <future>
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
  if (auto ctx = g_context.lock()) {
    if (!ctx->lost())
      return ctx;
    g_failed = true;
    g_failedHard = true;
    g_failReason = "the Vulkan device was lost";
    reason = g_failReason;
    hard = true;
    return nullptr;
  }
  if (g_failed) { // do not probe the driver again and again
    reason = g_failReason;
    hard = g_failedHard;
    return nullptr;
  }
  std::shared_ptr<VulkanContext> ctx(new VulkanContext);
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = ctx->init(reason, hard);
  if (debugEnabled())
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
  // Keep the context for the lifetime of the process: creating the Vulkan
  // instance and device takes 0.5-2 s, which applications that create
  // several ANARI devices would pay every time. It is deliberately never
  // destroyed (no device teardown while the process unloads the driver).
  static auto *keepAlive = new std::shared_ptr<VulkanContext>(ctx);
  (void)keepAlive;
  return ctx;
}

namespace {
std::future<void> *g_prefetch = nullptr;
std::once_flag g_prefetchOnce;
} // namespace

void VulkanContext::prefetch()
{
  std::call_once(g_prefetchOnce, []() {
    // never destroyed: the future of a running std::async blocks its destructor
    g_prefetch = new std::future<void>(std::async(std::launch::async, []() {
      std::string reason;
      bool hard = false;
      (void)VulkanContext::acquire(reason, hard);
    }));
  });
}

void VulkanContext::waitPrefetch()
{
  if (g_prefetch && g_prefetch->valid())
    g_prefetch->wait();
}

bool VulkanContext::createPipelines(std::string &reason)
{
  VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
  smi.codeSize = size_t(kRaytraceSpvSize);
  smi.pCode = reinterpret_cast<const uint32_t *>(kRaytraceSpv);
  VkDescriptorSetLayoutBinding bind[kNumBindings]{};
  for (int i = 0; i < kNumBindings; i++) {
    bind[i].binding = uint32_t(i);
    bind[i].descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
        : i == kNumBindings - 1     ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                    : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bind[i].descriptorCount = 1;
    bind[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo dsl{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  dsl.bindingCount = kNumBindings;
  dsl.pBindings = bind;
  VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
  pli.setLayoutCount = 1;
  if (vkCreateShaderModule(m_device, &smi, nullptr, &m_module) != VK_SUCCESS
      || vkCreateDescriptorSetLayout(m_device, &dsl, nullptr, &m_setLayout)
          != VK_SUCCESS) {
    reason = "could not create the shader module";
    return false;
  }
  pli.pSetLayouts = &m_setLayout;
  if (vkCreatePipelineLayout(m_device, &pli, nullptr, &m_pipeLayout)
      != VK_SUCCESS) {
    reason = "could not create the pipeline layout";
    return false;
  }
  const char *names[2] = {"genAabbs", "main"};
  VkPipeline *pipes[2] = {&m_aabbPipe, &m_tracePipe};
  for (int i = 0; i < 2; i++) {
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = m_module;
    ci.stage.pName = names[i];
    ci.layout = m_pipeLayout;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &ci, nullptr, pipes[i])
        != VK_SUCCESS) {
      reason = "could not create the compute pipelines";
      return false;
    }
  }
  return true;
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

  // TACHYON_VULKAN_DEVICE=<index>: only consider that device
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
    if (!createPipelines(reason)) {
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
    if (m_aabbPipe)
      vkDestroyPipeline(m_device, m_aabbPipe, nullptr);
    if (m_tracePipe)
      vkDestroyPipeline(m_device, m_tracePipe, nullptr);
    if (m_pipeLayout)
      vkDestroyPipelineLayout(m_device, m_pipeLayout, nullptr);
    if (m_setLayout)
      vkDestroyDescriptorSetLayout(m_device, m_setLayout, nullptr);
    if (m_module)
      vkDestroyShaderModule(m_device, m_module, nullptr);
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
  std::lock_guard<std::mutex> lock(m_allocMutex);
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
    // unified-memory devices (lavapipe, integrated GPUs) only expose
    // host-visible memory that is also device-local
    if (type == ~0u)
      type = findMemoryType(req.memoryTypeBits, hv, 0);
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

  b.size = req.size;
  b.hostCoherent = (m_mem.memoryTypes[type].propertyFlags & hc) != 0;
  if (kind != MemoryKind::DeviceLocal) {
    if (vkMapMemory(m_device, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped)
        != VK_SUCCESS)
      b.mapped = nullptr;
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
  std::lock_guard<std::mutex> lock(m_allocMutex);
  if (b.mapped)
    vkUnmapMemory(m_device, b.memory);
  vkDestroyBuffer(m_device, b.buffer, nullptr);
  vkFreeMemory(m_device, b.memory, nullptr);
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
  if (r == VK_ERROR_DEVICE_LOST)
    m_lost = true;
  if (r != VK_SUCCESS)
    throw std::runtime_error(r == VK_ERROR_DEVICE_LOST
            ? std::string("the Vulkan device was lost")
            : "Vulkan submit failed (" + std::to_string(int(r)) + ")");
}

} // namespace gpu
} // namespace tachyon_anari
