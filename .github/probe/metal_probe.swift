// Metal capability probe for CI. Prints what the runner's GPU reports and, when
// the device supports ray tracing, builds a one-triangle acceleration structure
// and runs an intersector query from a compute kernel. Always exits 0: the
// point is to print the capability, not to gate the build on it.
import Foundation
import Metal

func say(_ s: String) {
    print(s)
    fflush(stdout)
}

let kernelSource = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace metal::raytracing;

kernel void rt_probe(primitive_acceleration_structure accel [[buffer(0)]],
                     device float4 *out [[buffer(1)]],
                     uint tid [[thread_position_in_grid]])
{
    ray r;
    r.origin = float3(0.0f, 0.0f, 0.0f);
    // thread 0 looks at the triangle (z = -1), thread 1 looks away from it
    r.direction = (tid == 0) ? float3(0.0f, 0.0f, -1.0f) : float3(0.0f, 0.0f, 1.0f);
    r.min_distance = 0.0f;
    r.max_distance = 100.0f;
    intersector<triangle_data> isect;
    intersection_result<triangle_data> hit = isect.intersect(r, accel);
    out[tid] = float4(hit.type == intersection_type::triangle ? 1.0f : 0.0f,
                      hit.distance, float(hit.primitive_id), 0.0f);
}
"""

func runProbe() {
    say("== Metal capability probe")
    guard let device = MTLCreateSystemDefaultDevice() else {
        say("MTLCreateSystemDefaultDevice(): nil (no Metal device)")
        return
    }
    say("MTLCreateSystemDefaultDevice(): \(device.name)")
    say("device.name: \(device.name)")
    say("hasUnifiedMemory: \(device.hasUnifiedMemory)")
    if #available(macOS 13.0, *) {
        say("supportsRaytracing: \(device.supportsRaytracing)")
        say("supportsFunctionPointers: \(device.supportsFunctionPointers)")
    } else {
        say("supportsRaytracing / supportsFunctionPointers: not available on this macOS")
    }
    say("supportsFamily(Apple7): \(device.supportsFamily(.apple7))")
    say("supportsFamily(Apple8): \(device.supportsFamily(.apple8))")
    say("supportsFamily(Apple9): \(device.supportsFamily(.apple9))")

    guard #available(macOS 13.0, *), device.supportsRaytracing else {
        say("RT kernel: skipped (device does not report supportsRaytracing)")
        return
    }
    guard let queue = device.makeCommandQueue() else {
        say("RT kernel: no command queue")
        return
    }

    // One triangle at z = -1 that contains the point (0, 0, -1).
    let verts: [Float] = [-1, -1, -1,  1, -1, -1,  0, 1, -1]
    guard let vb = device.makeBuffer(bytes: verts,
                                     length: verts.count * MemoryLayout<Float>.stride,
                                     options: .storageModeShared) else {
        say("RT kernel: vertex buffer allocation failed")
        return
    }
    let geo = MTLAccelerationStructureTriangleGeometryDescriptor()
    geo.vertexBuffer = vb
    geo.vertexStride = 3 * MemoryLayout<Float>.stride
    geo.vertexFormat = .float3
    geo.triangleCount = 1
    geo.opaque = true
    let desc = MTLPrimitiveAccelerationStructureDescriptor()
    desc.geometryDescriptors = [geo]

    let sizes = device.accelerationStructureSizes(descriptor: desc)
    guard let accel = device.makeAccelerationStructure(size: sizes.accelerationStructureSize),
          let scratch = device.makeBuffer(length: sizes.buildScratchBufferSize,
                                          options: .storageModePrivate),
          let cb = queue.makeCommandBuffer(),
          let enc = cb.makeAccelerationStructureCommandEncoder() else {
        say("RT kernel: acceleration structure allocation failed")
        return
    }
    enc.build(accelerationStructure: accel, descriptor: desc,
              scratchBuffer: scratch, scratchBufferOffset: 0)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    say("AS build: status=\(cb.status.rawValue) size=\(sizes.accelerationStructureSize) bytes"
        + " error=\(cb.error?.localizedDescription ?? "none")")
    guard cb.status == .completed else { return }

    let lib: MTLLibrary
    do {
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        lib = try device.makeLibrary(source: kernelSource, options: opts)
    } catch {
        say("RT kernel: Metal source compile failed: \(error)")
        return
    }
    guard let fn = lib.makeFunction(name: "rt_probe") else {
        say("RT kernel: function rt_probe not found")
        return
    }
    do {
        let pso = try device.makeComputePipelineState(function: fn)
        guard let out = device.makeBuffer(length: 32, options: .storageModeShared),
              let cb2 = queue.makeCommandBuffer(),
              let ce = cb2.makeComputeCommandEncoder() else {
            say("RT kernel: dispatch allocation failed")
            return
        }
        ce.setComputePipelineState(pso)
        ce.setAccelerationStructure(accel, bufferIndex: 0)
        ce.setBuffer(out, offset: 0, index: 1)
        ce.useResource(accel, usage: .read)
        ce.useResource(vb, usage: .read)
        ce.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 2, height: 1, depth: 1))
        ce.endEncoding()
        cb2.commit()
        cb2.waitUntilCompleted()
        say("RT dispatch: status=\(cb2.status.rawValue) error=\(cb2.error?.localizedDescription ?? "none")")
        guard cb2.status == .completed else { return }

        let p = out.contents().bindMemory(to: Float.self, capacity: 8)
        say("ray 0 (toward triangle): hit=\(p[0]) t=\(p[1]) primitive=\(p[2])")
        say("ray 1 (away from triangle): hit=\(p[4]) t=\(p[5]) primitive=\(p[6])")
        let correct = p[0] == 1 && abs(p[1] - 1) < 1e-3 && p[2] == 0 && p[4] == 0
        say("RT kernel result: \(correct ? "CORRECT hit (t = 1, primitive 0) and clean miss" : "UNEXPECTED")")
    } catch {
        say("RT kernel: pipeline creation failed: \(error)")
    }
}

runProbe()
exit(0)
