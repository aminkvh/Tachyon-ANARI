// Copyright (c) 2026 Tachyon-ANARI contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled with ARC (-fobjc-arc).

#import "MetalContext.h"

#include "../../Backends.h"
#include "../shaders/raytrace.metal.h"

#include <chrono>
#include <cstdio>
#include <future>
#include <mutex>

// Everything Metal ray tracing needs is macOS 12+; acquire() checks that once
// at run time, nothing else runs without a context.
#pragma clang diagnostic ignored "-Wunguarded-availability"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

namespace tachyon_anari {
namespace gpu {

namespace {
std::mutex g_mutex;
std::weak_ptr<MetalContext> g_context;
bool g_failed{false};
bool g_failedHard{false};
std::string g_failReason;

std::string str(NSString *s)
{
  return s ? std::string([s UTF8String]) : std::string();
}

double msSince(std::chrono::steady_clock::time_point t0)
{
  return std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - t0)
      .count();
}

id<MTLLibrary> compileLibrary(id<MTLDevice> dev, std::string &error)
{
  NSString *src = [[NSString alloc] initWithBytes:kRaytraceMsl
                                           length:kRaytraceMslSize
                                         encoding:NSUTF8StringEncoding];
  if (!src) {
    error = "the embedded Metal source is not valid UTF-8";
    return nil;
  }
  MTLCompileOptions *opt = [[MTLCompileOptions alloc] init];
  // IEEE math like the Vulkan backend (fast math changes the results)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  opt.fastMathEnabled = NO;
#pragma clang diagnostic pop
  NSError *err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
  if (!lib)
    error = str(err.localizedDescription);
  return lib;
}

id<MTLFunction> traceFunction(id<MTLLibrary> lib)
{
  // slangc renames the entry point "main" (reserved in Metal)
  id<MTLFunction> fn = [lib newFunctionWithName:@"main_0"];
  if (!fn)
    fn = [lib newFunctionWithName:@"main"];
  return fn;
}
} // namespace

MetalContext::~MetalContext() = default;

bool MetalContext::init(std::string &reason, bool &hard)
{
  @autoreleasepool {
    hard = false;
    if (@available(macOS 12.0, *)) {
    } else {
      reason = "Metal ray tracing needs macOS 12 or newer";
      return false;
    }
    const bool dbg = debugEnabled();
    id<MTLDevice> def = MTLCreateSystemDefaultDevice();
    NSArray<id<MTLDevice>> *all = MTLCopyAllDevices();
    id<MTLDevice> chosen = nil;
    if (def && def.supportsRaytracing)
      chosen = def;
    std::string names;
    for (id<MTLDevice> d in all) {
      if (!names.empty())
        names += ", ";
      names += str(d.name);
      if (!chosen && d.supportsRaytracing)
        chosen = d;
    }
    if (!chosen) {
      if (!def && all.count == 0)
        reason = "no Metal device";
      else
        reason = (names.empty() ? str(def.name) : names)
            + " without ray tracing support";
      if (dbg && def) {
        // still check that the embedded source compiles (CI runners have no
        // ray tracing GPU)
        const auto t0 = std::chrono::steady_clock::now();
        std::string error;
        id<MTLLibrary> lib = compileLibrary(def, error);
        if (lib)
          std::fprintf(stderr,
              "[tachyon metal] shader source compiled on '%s' in %.0f ms "
              "(kernel %s; not used: no ray tracing)\n",
              [def.name UTF8String],
              msSince(t0),
              traceFunction(lib) ? "found" : "MISSING");
        else
          std::fprintf(stderr,
              "[tachyon metal] shader source did not compile on '%s': %s\n",
              [def.name UTF8String],
              error.c_str());
      }
      return false;
    }

    m_device = chosen;
    m_gpuName = str(chosen.name);
    m_queue = [chosen newCommandQueue];
    if (!m_queue) {
      reason = "cannot create a Metal command queue";
      hard = true;
      return false;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::string error;
    id<MTLLibrary> lib = compileLibrary(chosen, error);
    if (!lib) {
      reason = "Metal shader compilation failed: " + error;
      hard = true;
      return false;
    }
    const double tCompile = msSince(t0);
    id<MTLFunction> fn = traceFunction(lib);
    if (!fn) {
      reason = "the Metal shader has no tracing kernel";
      hard = true;
      return false;
    }
    NSError *err = nil;
    m_trace = [chosen newComputePipelineStateWithFunction:fn error:&err];
    if (!m_trace) {
      reason = "cannot create the Metal compute pipeline: "
          + str(err.localizedDescription);
      hard = true;
      return false;
    }
    // 8 x 8 like the Vulkan shader, fewer rows if the kernel needs that many
    // registers that the GPU cannot run 64 threads per group
    const NSUInteger maxThreads = m_trace.maxTotalThreadsPerThreadgroup;
    m_groupW = 8;
    m_groupH = maxThreads >= 64 ? 8 : (maxThreads >= 8 ? maxThreads / 8 : 1);
    if (maxThreads < 8)
      m_groupW = maxThreads > 0 ? maxThreads : 1;
    if (dbg)
      std::fprintf(stderr,
          "[tachyon metal] shader compiled in %.0f ms, pipeline in %.0f ms\n",
          tCompile,
          msSince(t0) - tCompile);
    return true;
  }
}

std::shared_ptr<MetalContext> MetalContext::acquire(
    std::string &reason, bool &hard)
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (auto ctx = g_context.lock())
    return ctx;
  if (g_failed) { // do not probe again and again
    reason = g_failReason;
    hard = g_failedHard;
    return nullptr;
  }
  std::shared_ptr<MetalContext> ctx(new MetalContext);
  const auto t0 = std::chrono::steady_clock::now();
  const bool ok = ctx->init(reason, hard);
  if (debugEnabled())
    std::fprintf(stderr,
        "[tachyon metal] context init %.0f ms%s%s\n",
        msSince(t0),
        ok ? "" : ": ",
        ok ? "" : reason.c_str());
  if (!ok) {
    g_failed = true;
    g_failReason = reason;
    g_failedHard = hard;
    return nullptr;
  }
  g_context = ctx;
  // Kept for the lifetime of the process (compiling the shader is not free)
  static auto *keepAlive = new std::shared_ptr<MetalContext>(ctx);
  (void)keepAlive;
  return ctx;
}

namespace {
std::future<void> *g_prefetch = nullptr;
std::once_flag g_prefetchOnce;
} // namespace

void MetalContext::prefetch()
{
  std::call_once(g_prefetchOnce, []() {
    // never destroyed: the future of a running std::async blocks its destructor
    g_prefetch = new std::future<void>(std::async(std::launch::async, []() {
      std::string reason;
      bool hard = false;
      (void)MetalContext::acquire(reason, hard);
    }));
  });
}

void MetalContext::waitPrefetch()
{
  if (g_prefetch && g_prefetch->valid())
    g_prefetch->wait();
}

} // namespace gpu
} // namespace tachyon_anari
