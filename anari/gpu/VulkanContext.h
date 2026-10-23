// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Process-wide Vulkan instance + device for the GPU backend: volk loads Vulkan
// at run time (so the device still works on machines without it), the first
// physical device with ray queries, acceleration structures and buffer device
// addresses is used. A tiny allocator (one VkDeviceMemory per buffer: the
// backend only has a handful of large buffers) replaces VMA.

#include <volk.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

namespace tachyon_anari {
namespace gpu {

struct GpuBuffer
{
  VkBuffer buffer{VK_NULL_HANDLE};
  VkDeviceMemory memory{VK_NULL_HANDLE};
  VkDeviceSize size{0};
  VkDeviceAddress address{0};
  void *mapped{nullptr};
  bool hostCoherent{true};
  bool deviceLocal{false};
  explicit operator bool() const { return buffer != VK_NULL_HANDLE; }
};

enum class MemoryKind
{
  DeviceLocal, // VRAM
  Upload, // host visible, coherent
  Readback // host visible + cached when available
};

class VulkanContext
{
 public:
  // Returns the shared context (created on first use, destroyed when the last
  // user releases it), or nullptr with 'reason' set. 'hard' tells a real
  // failure (driver error) from "no capable GPU / no Vulkan".
  static std::shared_ptr<VulkanContext> acquire(std::string &reason, bool &hard);

  ~VulkanContext();
  VulkanContext(const VulkanContext &) = delete;
  VulkanContext &operator=(const VulkanContext &) = delete;

  const std::string &gpuName() const { return m_gpuName; }
  VkDevice device() const { return m_device; }
  uint32_t scratchAlignment() const { return m_scratchAlign; }

  GpuBuffer createBuffer(
      VkDeviceSize size, VkBufferUsageFlags usage, MemoryKind kind);
  void destroyBuffer(GpuBuffer &b);
  // bytes of device-local memory currently allocated / peak
  size_t deviceBytes() const { return m_deviceBytes; }
  size_t peakDeviceBytes() const { return m_peakDeviceBytes; }
  void resetPeak() { m_peakDeviceBytes = m_deviceBytes; }
  // flush (host to device) or invalidate (device to host) a mapped buffer
  void syncMapped(GpuBuffer &b, bool toDevice);

  // Records a one-time command buffer with 'record' and waits for it.
  template <class F>
  void submitAndWait(F &&record)
  {
    VkCommandBuffer cb = beginCommands();
    record(cb);
    endAndWait(cb);
  }
  VkCommandBuffer beginCommands();
  void endAndWait(VkCommandBuffer cb);

 private:
  VulkanContext() = default;
  bool init(std::string &reason, bool &hard);
  uint32_t findMemoryType(uint32_t typeBits,
      VkMemoryPropertyFlags want,
      VkMemoryPropertyFlags avoid) const;

  VkInstance m_instance{VK_NULL_HANDLE};
  VkPhysicalDevice m_phys{VK_NULL_HANDLE};
  VkDevice m_device{VK_NULL_HANDLE};
  VkQueue m_queue{VK_NULL_HANDLE};
  uint32_t m_family{0};
  VkCommandPool m_pool{VK_NULL_HANDLE};
  VkFence m_fence{VK_NULL_HANDLE};
  VkPhysicalDeviceMemoryProperties m_mem{};
  uint32_t m_scratchAlign{128};
  std::string m_gpuName;
  size_t m_deviceBytes{0};
  size_t m_peakDeviceBytes{0};
};

} // namespace gpu
} // namespace tachyon_anari
