// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Process-wide Metal device for the Metal backend (Objective-C++ only): the
// system default device when it supports ray tracing, otherwise the first one
// that does. The tracing kernel is compiled from the embedded Metal source
// (generated from shaders/raytrace.slang) when the context is created, so no
// Xcode or offline shader compiler is needed.

#import <Metal/Metal.h>

#include <memory>
#include <string>

namespace tachyon_anari {
namespace gpu {

class MetalContext
{
 public:
  // Returns the shared context (created on first use and then kept for the
  // lifetime of the process), or nullptr with 'reason' set. 'hard' tells a
  // real failure (shader compilation, out of memory) from "no capable GPU".
  static std::shared_ptr<MetalContext> acquire(std::string &reason, bool &hard);

  // Starts creating the context on a background thread (compiling the shader
  // takes a moment); acquire() then only waits for the remainder.
  // waitPrefetch() blocks until that thread is done.
  static void prefetch();
  static void waitPrefetch();

  ~MetalContext();
  MetalContext(const MetalContext &) = delete;
  MetalContext &operator=(const MetalContext &) = delete;

  const std::string &gpuName() const { return m_gpuName; }
  id<MTLDevice> device() const { return m_device; }
  id<MTLCommandQueue> queue() const { return m_queue; }
  id<MTLComputePipelineState> tracePipeline() const { return m_trace; }
  // threadgroup size for the tracing kernel
  NSUInteger groupWidth() const { return m_groupW; }
  NSUInteger groupHeight() const { return m_groupH; }

 private:
  MetalContext() = default;
  bool init(std::string &reason, bool &hard);

  id<MTLDevice> m_device{nil};
  id<MTLCommandQueue> m_queue{nil};
  id<MTLComputePipelineState> m_trace{nil};
  NSUInteger m_groupW{8}, m_groupH{8};
  std::string m_gpuName;
};

} // namespace gpu
} // namespace tachyon_anari
