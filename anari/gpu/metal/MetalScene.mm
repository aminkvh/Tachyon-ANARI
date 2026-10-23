// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

// Compiled with ARC (-fobjc-arc).

#import "MetalContext.h"

#include "MetalScene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <dispatch/dispatch.h>

// Everything here runs only with a MetalContext, which exists only on macOS
// 12+ (checked at run time in MetalContext::init()).
#pragma clang diagnostic ignored "-Wunguarded-availability"
#pragma clang diagnostic ignored "-Wunguarded-availability-new"

namespace tachyon_anari {
namespace gpu {

namespace {

std::string str(NSString *s)
{
  return s ? std::string([s UTF8String]) : std::string();
}

double mb(size_t bytes)
{
  return double(bytes) / 1048576.0;
}

// Bounding boxes of spheres (4 floats each) or cylinders (8 floats each) as
// MTLAxisAlignedBoundingBox (min xyz, max xyz); same as genAabbs in
// raytrace.slang
void fillAabbs(float *out, const float *prims, uint32_t count, bool spheres)
{
  const size_t chunk = 65536;
  const size_t chunks = (size_t(count) + chunk - 1) / chunk;
  dispatch_apply(chunks,
      dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
      ^(size_t c) {
        const size_t end = std::min(size_t(count), (c + 1) * chunk);
        for (size_t i = c * chunk; i < end; i++) {
          float *o = out + 6 * i;
          if (spheres) {
            const float *s = prims + 4 * i;
            const float r = std::fabs(s[3]) * 1.00001f + 1e-6f;
            for (int k = 0; k < 3; k++) {
              o[k] = s[k] - r;
              o[3 + k] = s[k] + r;
            }
          } else {
            const float *a = prims + 8 * i;
            const float *b = a + 4;
            const float r = std::fabs(a[3]) * 1.00001f + 1e-6f;
            for (int k = 0; k < 3; k++) {
              o[k] = std::min(a[k], b[k]) - r;
              o[3 + k] = std::max(a[k], b[k]) + r;
            }
          }
        }
      });
}

} // namespace

struct MetalScene::Impl
{
  std::shared_ptr<MetalContext> ctx;
  id<MTLDevice> dev{nil};

  id<MTLBuffer> dummy{nil}; // bound for unused bindings
  id<MTLBuffer> frameBuf{nil}; // FrameData
  id<MTLBuffer> lightBuf{nil};
  size_t lightCapacity{0};

  // GPU scene
  id<MTLBuffer> buf[NUM_BINDINGS]; // by shader binding (1..18)
  id<MTLBuffer> aabbs[2]; // spheres, cylinders (kept for refits)
  std::vector<id<MTLAccelerationStructure>> blas; // per slice
  id<MTLAccelerationStructure> tlas{nil};
  size_t asBytes{0};

  // outputs
  id<MTLBuffer> outBuf{nil}, depthBuf{nil}, accumBuf{nil};
  int outW{0}, outH{0};
  bool outDepth{false};
  OutputFormat outFormat{OutputFormat::None};

  id<MTLBuffer> newBuffer(size_t bytes)
  {
    bytes = std::max<size_t>(bytes, 16);
    if (bytes > dev.maxBufferLength)
      throw std::runtime_error("a Metal buffer of "
          + std::to_string(size_t(mb(bytes))) + " MB exceeds the device limit");
    id<MTLBuffer> b =
        [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b)
      throw std::runtime_error("out of Metal memory ("
          + std::to_string(size_t(mb(bytes))) + " MB buffer)");
    return b;
  }

  id<MTLBuffer> scratchBuffer(NSUInteger bytes)
  {
    id<MTLBuffer> b = [dev newBufferWithLength:std::max<NSUInteger>(bytes, 16)
                                       options:MTLResourceStorageModePrivate];
    if (!b)
      throw std::runtime_error("out of Metal memory (scratch buffer)");
    return b;
  }

  // creates the buffer for 'binding' or, when it has the size, overwrites it
  // (the GPU is idle: every submission is waited for)
  void putBuffer(int binding, const void *data, size_t bytes)
  {
    __strong id<MTLBuffer> &b = buf[binding];
    if (!b || b.length != std::max<size_t>(bytes, 16))
      b = newBuffer(bytes);
    std::memcpy(b.contents, data, bytes);
  }

  // Records one command buffer with 'record', submits it and waits for it
  template <class F>
  void run(const char *what, F &&record)
  {
    std::string error;
    @autoreleasepool {
      id<MTLCommandBuffer> cb = [ctx->queue() commandBuffer];
      if (!cb) {
        error = "no command buffer";
      } else {
        record(cb);
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
          error = str(cb.error.localizedDescription);
          if (error.empty())
            error = "command buffer error";
        }
      }
    }
    if (!error.empty())
      throw std::runtime_error(std::string(what) + ": " + error);
  }

  id<MTLAccelerationStructure> newStructure(NSUInteger bytes)
  {
    id<MTLAccelerationStructure> as = [dev newAccelerationStructureWithSize:bytes];
    if (!as)
      throw std::runtime_error("out of Metal memory (acceleration structure of "
          + std::to_string(size_t(mb(bytes))) + " MB)");
    return as;
  }

  // Builds an acceleration structure; primitive ones that are not refitted
  // later are compacted
  id<MTLAccelerationStructure> build(
      MTLAccelerationStructureDescriptor *desc, bool compact)
  {
    MTLAccelerationStructureSizes sz =
        [dev accelerationStructureSizesWithDescriptor:desc];
    id<MTLAccelerationStructure> as = newStructure(sz.accelerationStructureSize);
    id<MTLBuffer> scratch = scratchBuffer(sz.buildScratchBufferSize);
    id<MTLBuffer> sizeBuf = compact ? newBuffer(16) : nil;
    run("building a Metal acceleration structure", [&](id<MTLCommandBuffer> cb) {
      id<MTLAccelerationStructureCommandEncoder> enc =
          [cb accelerationStructureCommandEncoder];
      [enc buildAccelerationStructure:as
                           descriptor:desc
                        scratchBuffer:scratch
                  scratchBufferOffset:0];
      if (sizeBuf)
        [enc writeCompactedAccelerationStructureSize:as
                                            toBuffer:sizeBuf
                                              offset:0];
      [enc endEncoding];
    });
    scratch = nil;
    if (!sizeBuf) {
      asBytes += sz.accelerationStructureSize;
      return as;
    }
    uint32_t compacted = 0;
    std::memcpy(&compacted, sizeBuf.contents, sizeof(compacted));
    if (compacted == 0 || compacted >= sz.accelerationStructureSize) {
      asBytes += sz.accelerationStructureSize;
      return as;
    }
    id<MTLAccelerationStructure> small = newStructure(compacted);
    run("compacting a Metal acceleration structure",
        [&](id<MTLCommandBuffer> cb) {
          id<MTLAccelerationStructureCommandEncoder> enc =
              [cb accelerationStructureCommandEncoder];
          [enc copyAndCompactAccelerationStructure:as
                           toAccelerationStructure:small];
          [enc endEncoding];
        });
    asBytes += compacted;
    return small;
  }

  // Refit: new primitive positions, same topology
  void refit(id<MTLAccelerationStructure> as,
      MTLPrimitiveAccelerationStructureDescriptor *desc)
  {
    MTLAccelerationStructureSizes sz =
        [dev accelerationStructureSizesWithDescriptor:desc];
    id<MTLBuffer> scratch = scratchBuffer(sz.refitScratchBufferSize);
    run("refitting a Metal acceleration structure", [&](id<MTLCommandBuffer> cb) {
      id<MTLAccelerationStructureCommandEncoder> enc =
          [cb accelerationStructureCommandEncoder];
      [enc refitAccelerationStructure:as
                           descriptor:desc
                          destination:nil
                        scratchBuffer:scratch
                  scratchBufferOffset:0];
      [enc endEncoding];
    });
  }
};

std::unique_ptr<MetalScene> MetalScene::create(
    std::string &gpuName, std::string &reason, bool &hard)
{
  auto ctx = MetalContext::acquire(reason, hard);
  if (!ctx)
    return nullptr;
  gpuName = ctx->gpuName();
  return std::unique_ptr<MetalScene>(new MetalScene(std::move(ctx)));
}

void MetalScene::prefetch()
{
  MetalContext::prefetch();
}

void MetalScene::waitPrefetch()
{
  MetalContext::waitPrefetch();
}

MetalScene::MetalScene(std::shared_ptr<MetalContext> ctx)
    : m(new Impl)
{
  m->ctx = std::move(ctx);
  m->dev = m->ctx->device();
  m->dummy = m->newBuffer(64);
  m->frameBuf = m->newBuffer(sizeof(FrameData));
}

MetalScene::~MetalScene() = default;

void MetalScene::destroyScene()
{
  for (int i = 0; i < NUM_BINDINGS; i++)
    m->buf[i] = nil;
  m->aabbs[0] = m->aabbs[1] = nil;
  m->blas.clear();
  m->tlas = nil;
  m->asBytes = 0;
  m_haveSig = false;
  m_updatable = false;
  m_refits = 0;
}

void MetalScene::destroyOutputs()
{
  m->outBuf = nil;
  m->depthBuf = nil;
  m->accumBuf = nil;
  m->outW = m->outH = 0;
}

void MetalScene::ensureLightBuffer(size_t lights)
{
  const size_t need = std::max<size_t>(lights, 1);
  if (m->lightBuf && m->lightCapacity >= need)
    return;
  m->lightBuf = m->newBuffer(32 * need);
  m->lightCapacity = need;
}

// Build ///////////////////////////////////////////////////////////////////////

void MetalScene::uploadSceneData(const Gathered &g)
{
  m->putBuffer(BIND_STYLES, g.styles.data(), 4 * g.styles.size());
  m->putBuffer(BIND_INSTANCES, g.instTable.data(), 4 * g.instTable.size());
  if (!g.samplerTable.empty()) {
    m->putBuffer(BIND_SAMPLERS, g.samplerTable.data(), 4 * g.samplerTable.size());
    const float zero[4] = {0, 0, 0, 0};
    m->putBuffer(BIND_TEXELS,
        g.texels.empty() ? zero : g.texels.data(),
        g.texels.empty() ? 16 : 4 * g.texels.size());
  }
  if (g.nSph) {
    m->putBuffer(BIND_SPHERES, g.spheres.data(), 16 * size_t(g.nSph));
    if (g.flags & F_SPHERE_COLORS)
      m->putBuffer(BIND_SPHERE_COLORS, g.sphereColors.data(), 12 * size_t(g.nSph));
    if (g.flags & F_SPHERE_STYLES)
      m->putBuffer(BIND_SPHERE_STYLES, g.sphereStyles.data(), 4 * size_t(g.nSph));
  }
  if (g.nCyl) {
    m->putBuffer(BIND_CYLS, g.cyls.data(), 32 * size_t(g.nCyl));
    if (g.flags & F_CYL_COLORS)
      m->putBuffer(BIND_CYL_COLORS, g.cylColors.data(), 24 * size_t(g.nCyl));
    if (g.flags & F_CYL_STYLES)
      m->putBuffer(BIND_CYL_STYLES, g.cylStyles.data(), 4 * size_t(g.nCyl));
  }
  if (g.nTri) {
    m->putBuffer(BIND_TRI_POS, g.triPos.data(), 36 * size_t(g.nTri));
    m->putBuffer(BIND_TRI_SHADE, g.triShade.data(), 48 * size_t(g.nTri));
    if (!g.triAux.empty())
      m->putBuffer(BIND_TRI_AUX, g.triAux.data(), 4 * g.triAux.size());
  }
}

// Builds (or with 'refitOnly' refits) the primitive acceleration structure of
// every slice
void MetalScene::buildSlices(const Gathered &g, bool updatable, bool refitOnly)
{
  if (!refitOnly)
    m->blas.assign(g.slices.size(), nil);
  auto finish = [&](size_t i, MTLAccelerationStructureGeometryDescriptor *geom) {
    MTLPrimitiveAccelerationStructureDescriptor *desc =
        [MTLPrimitiveAccelerationStructureDescriptor descriptor];
    desc.geometryDescriptors = @[ geom ];
    desc.usage = updatable ? MTLAccelerationStructureUsageRefit
                           : MTLAccelerationStructureUsageNone;
    if (refitOnly)
      m->refit(m->blas[i], desc);
    else
      m->blas[i] = m->build(desc, !updatable);
  };
  for (int type = 0; type < 2; type++) {
    const uint32_t total = type == 0 ? g.nSph : g.nCyl;
    if (!total)
      continue;
    const size_t bytes = 24 * size_t(total);
    __strong id<MTLBuffer> &boxes = m->aabbs[type];
    if (!boxes || boxes.length != bytes)
      boxes = m->newBuffer(bytes);
    fillAabbs(static_cast<float *>(boxes.contents),
        type == 0 ? g.spheres.data() : g.cyls.data(),
        total,
        type == 0);
    for (size_t i = 0; i < g.slices.size(); i++) {
      const Slice &s = g.slices[i];
      if (s.type != type)
        continue;
      @autoreleasepool {
        MTLAccelerationStructureBoundingBoxGeometryDescriptor *geom =
            [MTLAccelerationStructureBoundingBoxGeometryDescriptor descriptor];
        geom.boundingBoxBuffer = boxes;
        geom.boundingBoxBufferOffset = 24 * NSUInteger(s.base);
        geom.boundingBoxStride = 24;
        geom.boundingBoxCount = s.count;
        geom.opaque = YES;
        finish(i, geom);
      }
    }
    if (!updatable)
      boxes = nil; // only needed again for refits
  }
  for (size_t i = 0; i < g.slices.size(); i++) {
    const Slice &s = g.slices[i];
    if (s.type != 2)
      continue;
    @autoreleasepool {
      MTLAccelerationStructureTriangleGeometryDescriptor *geom =
          [MTLAccelerationStructureTriangleGeometryDescriptor descriptor];
      geom.vertexBuffer = m->buf[BIND_TRI_POS];
      geom.vertexBufferOffset = 36 * NSUInteger(s.base);
      geom.vertexStride = 12;
      // no index buffer: vertices 3 i .. 3 i + 2 form triangle i
      geom.triangleCount = s.count;
      geom.opaque = YES;
      finish(i, geom);
    }
  }
}

void MetalScene::buildTlas(const Gathered &g)
{
  m->tlas = nil;
  const size_t count = g.instances.size();
  id<MTLBuffer> instBuf = m->newBuffer(
      sizeof(MTLAccelerationStructureUserIDInstanceDescriptor) * count);
  auto *inst = static_cast<MTLAccelerationStructureUserIDInstanceDescriptor *>(
      instBuf.contents);
  for (size_t i = 0; i < count; i++) {
    const InstanceEntry &e = g.instances[i];
    MTLAccelerationStructureUserIDInstanceDescriptor &o = inst[i];
    std::memset(&o, 0, sizeof(o));
    // column major 4x3: columns 0..2 the linear part, 3 the translation
    for (int c = 0; c < 4; c++)
      for (int r = 0; r < 3; r++)
        o.transformationMatrix.columns[c].elements[r] = e.m[4 * r + c];
    o.options = MTLAccelerationStructureInstanceOptionDisableTriangleCulling;
    o.mask = 0xFF;
    o.intersectionFunctionTableOffset = 0;
    o.accelerationStructureIndex = uint32_t(e.slice);
    o.userID = uint32_t(i); // index of the record in the instance table
  }
  @autoreleasepool {
    NSMutableArray<id<MTLAccelerationStructure>> *structures =
        [NSMutableArray arrayWithCapacity:m->blas.size()];
    for (id<MTLAccelerationStructure> as : m->blas)
      [structures addObject:as];
    MTLInstanceAccelerationStructureDescriptor *desc =
        [MTLInstanceAccelerationStructureDescriptor descriptor];
    desc.instancedAccelerationStructures = structures;
    desc.instanceCount = count;
    desc.instanceDescriptorBuffer = instBuf;
    desc.instanceDescriptorBufferOffset = 0;
    desc.instanceDescriptorStride =
        sizeof(MTLAccelerationStructureUserIDInstanceDescriptor);
    desc.instanceDescriptorType =
        MTLAccelerationStructureInstanceDescriptorTypeUserID;
    m->tlas = m->build(desc, false);
  }
}

void MetalScene::buildFull(Gathered &g, bool updatable)
{
  const double t0 = nowMs();
  destroyScene();
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
  if (m_debug)
    std::fprintf(stderr,
        "[tachyon metal] build: %u spheres, %u cylinders, %u triangles in %zu "
        "BLAS, %zu TLAS instances, %zu styles, %zu samplers, %zu lights: "
        "upload %.1f ms, acceleration structures %.1f ms (%.1f MB)%s\n",
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
        mb(m->asBytes),
        updatable ? " (refittable)" : "");
}

void MetalScene::refit(Gathered &g)
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
        "[tachyon metal] refit %u: upload %.1f ms, BLAS refit + TLAS %.1f ms\n",
        m_refits,
        t1 - t0,
        nowMs() - t1);
}

void MetalScene::finishBuild()
{
  Gathered g = gather();
  m_placements.clear();
  if (g.nSph + g.nCyl + g.nTri == 0) {
    destroyScene();
    return;
  }
  const bool same = m->tlas && m_haveSig && g.sig == m_sig;
  try {
    if (same && m_updatable && m_refits < m_maxRefits) {
      refit(g);
    } else {
      // The second build with the same topology in a row: the scene is being
      // animated, so make the structures refittable from now on. Also after
      // the maximum number of refits (refitting degrades the quality).
      const bool animate = same || (m_haveSig && m_updatable);
      buildFull(g, animate);
    }
  } catch (...) {
    destroyScene();
    throw;
  }
}

// Frame ///////////////////////////////////////////////////////////////////////

void MetalScene::render()
{
  if (m_width <= 0 || m_height <= 0)
    return;
  if (!m->tlas) { // nothing renderable
    fillDirectBackground();
    return;
  }
  const double t0 = nowMs();
  FrameSetup setup = prepareFrame();
  const size_t np = setup.numPixels;
  const bool wantDepth = setup.depth;
  const OutputFormat fmt = setup.format;
  const size_t pixBytes = setup.pixelBytes;

  // Resources -----------------------------------------------------------------
  if (m->outW != m_width || m->outH != m_height || m->outDepth != wantDepth
      || m->outFormat != fmt || (setup.accumulate && !m->accumBuf)) {
    const bool keepAccum = setup.accumulate && m->accumBuf
        && m->outW == m_width && m->outH == m_height;
    id<MTLBuffer> keep = m->accumBuf;
    destroyOutputs();
    if (keepAccum)
      m->accumBuf = keep;
    keep = nil;
    m->outBuf = m->newBuffer(pixBytes);
    if (wantDepth)
      m->depthBuf = m->newBuffer(4 * np);
    if (setup.accumulate && !m->accumBuf)
      m->accumBuf = m->newBuffer(12 * np);
    m->outW = m_width;
    m->outH = m_height;
    m->outDepth = wantDepth;
    m->outFormat = fmt;
  }
  std::memcpy(m->frameBuf.contents, &setup.fd, sizeof(FrameData));
  ensureLightBuffer(setup.lights.size() / 8);
  std::memcpy(
      m->lightBuf.contents, setup.lights.data(), 4 * setup.lights.size());

  id<MTLBuffer> bind[NUM_BINDINGS];
  for (int i = 0; i < NUM_BINDINGS; i++)
    bind[i] = m->buf[i];
  bind[BIND_LIGHTS] = m->lightBuf;
  bind[BIND_OUT] = m->outBuf;
  bind[BIND_DEPTH] = m->depthBuf;
  bind[BIND_ACCUM] = m->accumBuf;
  bind[BIND_FRAME] = m->frameBuf;
  const NSUInteger gw = m->ctx->groupWidth(), gh = m->ctx->groupHeight();
  m->run("rendering a Metal frame", [&](id<MTLCommandBuffer> cb) {
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:m->ctx->tracePipeline()];
    for (int i = BIND_SPHERES; i < NUM_BINDINGS; i++)
      [enc setBuffer:(bind[i] ? bind[i] : m->dummy) offset:0 atIndex:NSUInteger(i)];
    [enc setAccelerationStructure:m->tlas atBufferIndex:BIND_AS];
    // the primitive structures are only referenced through the instance one
    for (id<MTLAccelerationStructure> as : m->blas)
      [enc useResource:as usage:MTLResourceUsageRead];
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger(m_width) + gw - 1) / gw,
                                  (NSUInteger(m_height) + gh - 1) / gh,
                                  1)
        threadsPerThreadgroup:MTLSizeMake(gw, gh, 1)];
    [enc endEncoding];
  });
  const double t1 = nowMs();

  if (fmt != OutputFormat::None)
    std::memcpy(m_colorDst, m->outBuf.contents, pixBytes);
  if (wantDepth)
    std::memcpy(m_depthDst, m->depthBuf.contents, 4 * np);
  if (m_debug)
    std::fprintf(stderr,
        "[tachyon metal] frame %dx%d: trace %.2f ms, readback %.2f ms"
        " (samples %u+%u)\n",
        m_width,
        m_height,
        t1 - t0,
        nowMs() - t1,
        setup.base,
        setup.count);
}

} // namespace gpu
} // namespace tachyon_anari
