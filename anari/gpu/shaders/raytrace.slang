// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

// Phase 1 (spike) shaders of the Vulkan backend: spheres only, eyeLight
// shading. Compiled to SPIR-V by slangc (see anari/gpu/CMakeLists.txt and
// TACHYON_REGENERATE_SHADERS); the result is committed as raytrace.spv.h.
//
//   slangc raytrace.slang -target spirv -profile spirv_1_4 \
//     -capability spvRayQueryKHR -entry genAabbs -stage compute \
//     -entry main -stage compute -o raytrace.spv
//
// The same source is meant to be compiled to Metal later (-target metal).

struct PushConstants
{
    float4 posOrtho; // camera position, w: 1 = orthographic
    float4 fwdU0; // forward, w: image plane u of the first pixel center
    float4 rightDu; // right, w: u step per pixel
    float4 upV0; // up, w: image plane v of the first (top) pixel center
    float4 shade; // x: ambient weight, y: diffuse weight, z: v step per pixel
    float4 bg; // background color
    float4 baseColor; // color of all spheres when there is no color buffer
    uint2 size; // image size
    uint2 flags; // x: color buffer present, y: write depth
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] StructuredBuffer<float4> spheres; // xyz center, w radius
[[vk::binding(2, 0)]] StructuredBuffer<float> colors; // 3 floats per sphere
[[vk::binding(3, 0)]] RWStructuredBuffer<float> outRgb; // 3 floats per pixel
[[vk::binding(4, 0)]] RWStructuredBuffer<float> outDepth;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> aabbs; // 6 floats per sphere

[shader("compute")]
[numthreads(64, 1, 1)]
void genAabbs(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i >= pc.size.x)
        return;
    float4 s = spheres[i];
    float r = abs(s.w) * 1.00001 + 1e-6;
    aabbs[6 * i + 0] = s.x - r;
    aabbs[6 * i + 1] = s.y - r;
    aabbs[6 * i + 2] = s.z - r;
    aabbs[6 * i + 3] = s.x + r;
    aabbs[6 * i + 4] = s.y + r;
    aabbs[6 * i + 5] = s.z + r;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= pc.size.x || tid.y >= pc.size.y)
        return;

    float u = pc.fwdU0.w + pc.rightDu.w * (float(tid.x) + 0.5);
    float v = pc.upV0.w + pc.shade.z * (float(tid.y) + 0.5);
    float3 fwd = pc.fwdU0.xyz;
    float3 plane = pc.rightDu.xyz * u + pc.upV0.xyz * v;

    RayDesc ray;
    if (pc.posOrtho.w > 0.5)
    {
        ray.Origin = pc.posOrtho.xyz + plane;
        ray.Direction = fwd;
    }
    else
    {
        ray.Origin = pc.posOrtho.xyz;
        ray.Direction = normalize(fwd + plane);
    }
    ray.TMin = 0.0;
    ray.TMax = 1e30;

    float best = ray.TMax;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            float4 s = spheres[q.CandidatePrimitiveIndex()];
            float3 oc = ray.Origin - s.xyz;
            float b = dot(oc, ray.Direction);
            float3 h = oc - b * ray.Direction; // closest approach: stable
            float disc = s.w * s.w - dot(h, h);
            if (disc >= 0.0)
            {
                float sq = sqrt(disc);
                float t = -b - sq;
                if (t <= 0.0)
                    t = -b + sq;
                if (t > 0.0 && t < best)
                {
                    best = t;
                    q.CommitProceduralPrimitiveHit(t);
                }
            }
        }
    }

    float3 rgb = pc.bg.xyz;
    float depth = 1e18;
    if (q.CommittedStatus() == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
    {
        uint prim = q.CommittedPrimitiveIndex();
        float4 s = spheres[prim];
        float3 p = ray.Origin + best * ray.Direction;
        float3 n = (p - s.xyz) / s.w;
        // like the CPU backend: the headlight shines along the camera
        // direction (not per ray), the normal faces the viewer
        if (dot(n, ray.Direction) > 0.0)
            n = -n;
        float ndl = max(dot(n, -fwd), 0.0);
        float3 col = pc.baseColor.xyz;
        if (pc.flags.x != 0u)
            col = float3(colors[3 * prim], colors[3 * prim + 1], colors[3 * prim + 2]);
        rgb = col * (pc.shade.x + pc.shade.y * ndl);
        depth = best;
    }

    uint idx = tid.y * pc.size.x + tid.x;
    outRgb[3 * idx + 0] = rgb.x;
    outRgb[3 * idx + 1] = rgb.y;
    outRgb[3 * idx + 2] = rgb.z;
    if (pc.flags.y != 0u)
        outDepth[idx] = depth;
}
