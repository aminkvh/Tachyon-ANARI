// Copyright (c) 2026 Tachyon-dev contributors
// SPDX-License-Identifier: Apache-2.0

// Shaders of the GPU backends. Compiled by slangc (see anari/CMakeLists.txt
// and TACHYON_REGENERATE_SHADERS); the results are committed as raytrace.spv.h
// (Vulkan) and raytrace.metal.h (Metal source, compiled when the device
// starts):
//
//   slangc raytrace.slang -target spirv -profile spirv_1_4 \
//     -capability spvRayQueryKHR -entry genAabbs -stage compute \
//     -entry main -stage compute -o raytrace.spv
//   slangc raytrace.slang -target metal -DTACHYON_METAL \
//     -line-directive-mode none -entry main -stage compute -o raytrace.metal
//
// The shading is a port of Tachyon's full shader (src/shade.c) as the CPU
// backend drives it: constant ambient + sum of Lambertian lights (with
// shadow rays) times the surface color, Blinn highlights, transparency by
// continuing the ray (RT_TRANS_ORIG). eyeLight and the unlit default mode are
// the same code with one headlight. Code that only the Metal output needs is
// under TACHYON_METAL, so the SPIR-V does not change with it. The Metal
// backend computes the bounding boxes on the CPU (genAabbs is Vulkan only).

static const float MINCONTRIB = 0.001959;

struct FrameData
{
    float4 posOrtho; // camera position, w: 1 = orthographic
    float4 fwdU0; // forward, w: image plane u of the first pixel (corner)
    float4 rightDu; // right, w: u step per pixel
    float4 upV0; // up, w: image plane v of the first (top) pixel (corner)
    float4 shade; // x: ambient weight, y: diffuse scale, z: v step per pixel
    float4 bg; // background color, w: alpha
    float4 unused0;
    uint4 size; // x,y: image size, z: samples per pixel, w: first sample index
    uint4 counts; // spheres, cylinders, triangles, lights
    // x: flags (see below), y: output format, z: max ray depth,
    // w: primitive set to generate bounding boxes for
    uint4 flags;
    uint4 styles; // uniform style id of spheres, of cylinders
};

static const uint F_SPHERE_COLORS = 1u;
static const uint F_SPHERE_STYLES = 2u;
static const uint F_CYL_COLORS = 4u;
static const uint F_CYL_STYLES = 8u;
static const uint F_SHADOWS = 16u;
static const uint F_DEPTH = 32u;
static const uint F_ACCUM = 64u; // the accumulation buffer exists
static const uint F_ACCUM_READ = 128u; // sum in it from previous frames
static const uint F_BG_ALPHA = 256u; // background alpha < 1
static const uint F_TRANSPARENT = 512u; // some style has opacity < 1

static const uint FMT_NONE = 0u;
static const uint FMT_U8 = 1u;
static const uint FMT_U8_SRGB = 2u;
static const uint FMT_F32 = 3u;

[[vk::binding(0, 0)]] RaytracingAccelerationStructure tlas;
[[vk::binding(1, 0)]] StructuredBuffer<float4> spheres; // xyz center, w radius
[[vk::binding(2, 0)]] StructuredBuffer<float> sphereColors; // 3 per sphere
[[vk::binding(3, 0)]] StructuredBuffer<uint> sphereStyles;
[[vk::binding(4, 0)]] StructuredBuffer<float4> cylinders; // 2 per cylinder
[[vk::binding(5, 0)]] StructuredBuffer<float> cylColors; // 6 per cylinder
[[vk::binding(6, 0)]] StructuredBuffer<uint> cylStyles;
[[vk::binding(7, 0)]] StructuredBuffer<float> triPos; // 9 per triangle
// 12 per triangle: 3 normals, style, flags, aux offset (uints as floats)
[[vk::binding(8, 0)]] StructuredBuffer<float> triShade;
[[vk::binding(9, 0)]] StructuredBuffer<float> triAux; // vertex colors / attributes
// 2 per style: (rgb, opacity), (diffuse, phong, phong exponent, metal)
[[vk::binding(10, 0)]] StructuredBuffer<float4> styleTable;
[[vk::binding(11, 0)]] StructuredBuffer<float4> samplerTable; // 13 per sampler
[[vk::binding(12, 0)]] StructuredBuffer<float4> texels;
// 2 per light: (direction to the light | position, 0 directional / 1 point),
// (color, 0)
[[vk::binding(13, 0)]] StructuredBuffer<float4> lights;
[[vk::binding(14, 0)]] RWStructuredBuffer<uint> outPixels;
[[vk::binding(15, 0)]] RWStructuredBuffer<float> outDepth;
[[vk::binding(16, 0)]] RWStructuredBuffer<float> accum; // 3 per pixel
[[vk::binding(17, 0)]] RWStructuredBuffer<float> aabbs; // 6 per primitive
// TLAS instance records, see the Instances section below
[[vk::binding(18, 0)]] StructuredBuffer<float4> instances;
[[vk::binding(19, 0)]] ConstantBuffer<FrameData> fd;

[shader("compute")]
[numthreads(64, 1, 1)]
void genAabbs(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x + tid.y * fd.styles.z; // 2D dispatch for large counts
    if (fd.flags.w == 0u)
    {
        if (i >= fd.counts.x)
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
    else
    {
        if (i >= fd.counts.y)
            return;
        float4 a = cylinders[2 * i];
        float4 b = cylinders[2 * i + 1];
        float r = abs(a.w) * 1.00001 + 1e-6;
        float3 lo = min(a.xyz, b.xyz) - r;
        float3 hi = max(a.xyz, b.xyz) + r;
        aabbs[6 * i + 0] = lo.x;
        aabbs[6 * i + 1] = lo.y;
        aabbs[6 * i + 2] = lo.z;
        aabbs[6 * i + 3] = hi.x;
        aabbs[6 * i + 4] = hi.y;
        aabbs[6 * i + 5] = hi.z;
    }
}

// Primitives ///////////////////////////////////////////////////////////////

// Nearest intersection of the ray with sphere 'i' in (tmin, tmax)
bool hitSphere(uint i, float3 o, float3 d, float tmin, float tmax, out float t)
{
    t = 0.0;
    float4 s = spheres[i];
    float3 oc = o - s.xyz;
    float b = dot(oc, d);
    float3 h = oc - b * d; // closest approach: numerically stable
    float disc = s.w * s.w - dot(h, h);
    if (disc < 0.0)
        return false;
    float sq = sqrt(disc);
    float t0 = -b - sq;
    if (t0 > tmin && t0 < tmax)
    {
        t = t0;
        return true;
    }
    float t1 = -b + sq;
    if (t1 > tmin && t1 < tmax)
    {
        t = t1;
        return true;
    }
    return false;
}

// Finite cylinder with optional flat caps (bit 0: at p0, bit 1: at p1).
// Returns the nearest hit in (tmin, tmax); 'n' is the outward normal and
// 'sel' the axial position in [0, 1] of the hit (selects the half's color).
bool hitCylinder(uint i, float3 o, float3 d, float tmin, float tmax,
    out float t, out float3 n, out float sel)
{
    t = 0.0;
    n = float3(0, 0, 1);
    sel = 0.0;
    float4 a = cylinders[2 * i];
    float4 b = cylinders[2 * i + 1];
    uint caps = asuint(b.w);
    float3 p0 = a.xyz;
    float r = a.w;
    float3 axis = b.xyz - p0;
    float len = length(axis);
    if (!(len > 0.0))
        return false;
    float3 u = axis / len;
    float3 oc = o - p0;
    float ocu = dot(oc, u);
    float du = dot(d, u);
    float3 dp = d - u * du;
    float3 op = oc - u * ocu;

    float best = tmax;
    bool found = false;
    float A = dot(dp, dp);
    if (A > 1e-20)
    {
        float B = dot(op, dp);
        float C = dot(op, op) - r * r;
        float disc = B * B - A * C;
        if (disc >= 0.0)
        {
            float sq = sqrt(disc);
            float t0 = (-B - sq) / A;
            float t1 = (-B + sq) / A;
            [unroll]
            for (int k = 0; k < 2; k++)
            {
                float tt = k == 0 ? t0 : t1;
                if (tt > tmin && tt < best)
                {
                    float s = ocu + tt * du;
                    if (s >= 0.0 && s <= len)
                    {
                        best = tt;
                        found = true;
                        float3 radial = op + tt * dp;
                        n = radial / r;
                        sel = s / len;
                    }
                }
            }
        }
    }
    if (abs(du) > 1e-12)
    {
        [unroll]
        for (int c = 0; c < 2; c++)
        {
            if ((caps & (1u << c)) == 0u)
                continue;
            float plane = c == 0 ? 0.0 : len;
            float tt = (plane - ocu) / du;
            if (tt > tmin && tt < best)
            {
                float3 radial = op + tt * dp;
                if (dot(radial, radial) <= r * r)
                {
                    best = tt;
                    found = true;
                    n = c == 0 ? -u : u;
                    sel = c == 0 ? 0.0 : 1.0;
                }
            }
        }
    }
    t = best;
    return found;
}

// Instances ///////////////////////////////////////////////////////////////////

// Every TLAS instance has a record in "instances" (4 float4):
//   (type, first primitive of its slice, flags, unused), then the rows of the
//   object-to-world matrix (3x3 in xyz, translation in w).
// type: 0 spheres, 1 cylinders, 2 triangles. flags: 1 identity transform,
// 2 negative determinant. The transforms are rotations, translations and
// uniform scales, so normals transform like directions.
struct InstInfo
{
    uint type;
    uint base;
    uint flags;
    float4 r0;
    float4 r1;
    float4 r2;
};

InstInfo loadInst(uint i)
{
    InstInfo f;
    uint4 hdr = asuint(instances[4u * i]);
    f.type = hdr.x;
    f.base = hdr.y;
    f.flags = hdr.z;
    f.r0 = instances[4u * i + 1u];
    f.r1 = instances[4u * i + 2u];
    f.r2 = instances[4u * i + 3u];
    return f;
}

float3 rotate(InstInfo f, float3 v)
{
    return float3(dot(f.r0.xyz, v), dot(f.r1.xyz, v), dot(f.r2.xyz, v));
}

float3 toWorld(InstInfo f, float3 p)
{
    return rotate(f, p) + float3(f.r0.w, f.r1.w, f.r2.w);
}

// ray into object space (inverse of a similarity: transpose / scale^2)
void toObjectRay(InstInfo f, float3 o, float3 d, out float3 oo, out float3 od)
{
    float s2 = dot(f.r0.xyz, f.r0.xyz);
    float3 q = o - float3(f.r0.w, f.r1.w, f.r2.w);
    oo = (f.r0.xyz * q.x + f.r1.xyz * q.y + f.r2.xyz * q.z) / s2;
    od = (f.r0.xyz * d.x + f.r1.xyz * d.y + f.r2.xyz * d.z) / s2;
}

// Closest hit ///////////////////////////////////////////////////////////////

// Intersects a candidate primitive given the object space ray the hardware
// computed for its instance; t is the same ray parameter as in world space.
bool procHit(uint inst, uint localPrim, float3 oo, float3 od, float tmin,
    float tmax, out float t)
{
    uint4 hdr = asuint(instances[4u * inst]);
    uint gi = hdr.y + localPrim;
    float dl = length(od);
    float3 dn = od / dl;
    float tn;
    bool hit;
    if (hdr.x == 0u)
        hit = hitSphere(gi, oo, dn, tmin * dl, tmax * dl, tn);
    else
    {
        float3 n;
        float sel;
        hit = hitCylinder(gi, oo, dn, tmin * dl, tmax * dl, tn, n, sel);
    }
    t = tn / dl;
    return hit;
}

struct Hit
{
    bool valid;
    uint inst; // instance record
    uint prim; // primitive within the instance's slice
    float t;
    float2 bary;
};

Hit closestHit(float3 o, float3 d, float tmin, float tmax)
{
    Hit h;
    h.valid = false;
    h.inst = 0u;
    h.prim = 0u;
    h.t = tmax;
    h.bary = float2(0, 0);

    RayDesc ray;
    ray.Origin = o;
    ray.Direction = d;
    ray.TMin = tmin;
    ray.TMax = tmax;
#if defined(TACHYON_METAL)
    // Metal: triangles are reported as candidates too (forced non-opaque) and
    // committed here, so "best" always is the committed distance and a sphere
    // or cylinder behind a committed triangle is never committed over it.
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);
    float best = ray.TMax;
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            float t = q.CandidateTriangleRayT();
            if (t > tmin && t < best)
            {
                best = t;
                q.CommitNonOpaqueTriangleHit();
            }
        }
        else if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
#else
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(tlas, RAY_FLAG_NONE, 0xFF, ray);
    float best = ray.TMax;
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
#endif
            float t;
            if (procHit(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(),
                    q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(),
                    tmin, best, t)
                && t < best)
            {
                best = t;
                q.CommitProceduralPrimitiveHit(t);
            }
        }
    }
    uint status = q.CommittedStatus();
    if (status == COMMITTED_PROCEDURAL_PRIMITIVE_HIT)
    {
        h.valid = true;
        h.inst = q.CommittedInstanceID();
        h.prim = q.CommittedPrimitiveIndex();
        h.t = best;
    }
    else if (status == COMMITTED_TRIANGLE_HIT)
    {
        h.valid = true;
        h.inst = q.CommittedInstanceID();
        h.prim = q.CommittedPrimitiveIndex();
        h.t = q.CommittedRayT();
        h.bary = q.CommittedTriangleBarycentrics();
    }
    return h;
}

// True when anything blocks the ray before "tmax"
bool occluded(float3 o, float3 d, float tmin, float tmax)
{
    RayDesc ray;
    ray.Origin = o;
    ray.Direction = d;
    ray.TMin = tmin;
    ray.TMax = tmax;
#if defined(TACHYON_METAL)
    // Slang's Metal output ignores RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH:
    // stop at the first blocking candidate instead (triangles are forced
    // non-opaque so that they are reported here as well).
    RayQuery<RAY_FLAG_FORCE_NON_OPAQUE> q;
    q.TraceRayInline(tlas, RAY_FLAG_FORCE_NON_OPAQUE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            float t = q.CandidateTriangleRayT();
            if (t > tmin && t < tmax)
                return true;
        }
        else if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            float t;
            if (procHit(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(),
                    q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(),
                    tmin, tmax, t))
                return true;
        }
    }
    return false;
#else
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_PROCEDURAL_PRIMITIVE)
        {
            float t;
            if (procHit(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(),
                    q.CandidateObjectRayOrigin(), q.CandidateObjectRayDirection(),
                    tmin, tmax, t))
                q.CommitProceduralPrimitiveHit(t);
        }
    }
    return q.CommittedStatus() != COMMITTED_NOTHING;
#endif
}

// Samplers ///////////////////////////////////////////////////////////////////

float4 mulMat(float4 c0, float4 c1, float4 c2, float4 c3, float4 v)
{
    return c0 * v.x + c1 * v.y + c2 * v.z + c3 * v.w;
}

// wrap an integer texel index like helium::math::calculateWrapIndex
int wrapIndex(int i, uint size, uint mode)
{
    int n = int(size);
    if (mode == 1u)
    {
        // C++ evaluates i % size with size_t: a negative i is first converted
        // to a huge unsigned number
        if (i >= 0)
            return i % n;
        uint a = (0xFFFFFFFFu % size + 1u) % size; // 2^32 mod size
        uint r64 = (a * a) % size; // 2^64 mod size
        uint k = uint(-i) % size;
        return int((r64 + size - k) % size);
    }
    if (mode == 2u)
    {
        int x = i + (i < 0 ? 1 : 0);
        x = abs(x) % (2 * n);
        return x >= n ? 2 * n - x - 1 : x;
    }
    return clamp(i, 0, n - 1);
}

float4 texel(uint base, uint3 dims, int3 i, uint3 wrap)
{
    int x = wrapIndex(i.x, dims.x, wrap.x);
    int y = wrapIndex(i.y, dims.y, wrap.y);
    int z = wrapIndex(i.z, dims.z, wrap.z);
    return texels[base + (uint(z) * dims.y + uint(y)) * dims.x + uint(x)];
}

float4 evalSampler(uint sid, float4 attr)
{
    uint b = 13u * sid;
    uint4 hdr0 = asuint(samplerTable[b + 0]); // type, dims
    float4 hdr1 = samplerTable[b + 1]; // wraps, linear
    uint4 hdr2 = asuint(samplerTable[b + 2]); // texel base
    float4 outv;
    if (hdr0.x == 0u)
    {
        outv = attr;
    }
    else
    {
        float4 co = mulMat(samplerTable[b + 3], samplerTable[b + 4],
                           samplerTable[b + 5], samplerTable[b + 6], attr)
                    + samplerTable[b + 7];
        uint3 dims = uint3(hdr0.y, hdr0.z, hdr0.w);
        uint3 wrap = uint3(asuint(hdr1.x), asuint(hdr1.y), asuint(hdr1.z));
        bool linearFilter = asuint(hdr1.w) != 0u;
        uint dimCount = hdr0.x; // 1, 2 or 3
        float cc[3] = { co.x, co.y, co.z };
        int lo[3] = { 0, 0, 0 };
        int hi[3] = { 0, 0, 0 };
        float fr[3] = { 0, 0, 0 };
        for (uint a = 0; a < 3; a++)
        {
            if (a >= dimCount)
                continue;
            float c = cc[a];
            if (wrap[a] == 1u)
                c = c - floor(c);
            float scale = float(dims[a]);
            float lowf = (c - (0.5 / scale)) * scale;
            float fl = floor(lowf);
            lo[a] = int(fl);
            hi[a] = lo[a] + 1;
            fr[a] = lowf - fl;
        }
        uint tb = hdr2.x;
        if (!linearFilter)
        {
            int3 pick = int3(fr[0] < 0.5 ? lo[0] : hi[0],
                             fr[1] < 0.5 ? lo[1] : hi[1],
                             fr[2] < 0.5 ? lo[2] : hi[2]);
            outv = texel(tb, dims, pick, wrap);
        }
        else
        {
            float4 acc = float4(0, 0, 0, 0);
            for (uint cz = 0; cz < (dimCount > 2u ? 2u : 1u); cz++)
            for (uint cy = 0; cy < (dimCount > 1u ? 2u : 1u); cy++)
            for (uint cx = 0; cx < 2u; cx++)
            {
                float wgt = (cx == 0u ? 1.0 - fr[0] : fr[0])
                          * (dimCount > 1u ? (cy == 0u ? 1.0 - fr[1] : fr[1]) : 1.0)
                          * (dimCount > 2u ? (cz == 0u ? 1.0 - fr[2] : fr[2]) : 1.0);
                int3 idx = int3(cx == 0u ? lo[0] : hi[0],
                                cy == 0u ? lo[1] : hi[1],
                                cz == 0u ? lo[2] : hi[2]);
                acc += wgt * texel(tb, dims, idx, wrap);
            }
            outv = acc;
        }
    }
    // output transform and offset
    return mulMat(samplerTable[b + 8], samplerTable[b + 9],
                  samplerTable[b + 10], samplerTable[b + 11], outv)
           + samplerTable[b + 12];
}

uint primStyle(Hit h)
{
    InstInfo info = loadInst(h.inst);
    uint gi = info.base + h.prim;
    if (info.type == 0u)
        return (fd.flags.x & F_SPHERE_STYLES) != 0u ? sphereStyles[gi] : fd.styles.x;
    if (info.type == 1u)
        return (fd.flags.x & F_CYL_STYLES) != 0u ? cylStyles[gi] : fd.styles.y;
    return asuint(triShade[12u * gi + 9u]);
}

// Light left after the shadow ray: opaque surfaces (opacity >= 0.99999) block
// it, transparent ones scale it by (1 - opacity), like Tachyon's shadow
// filtering.
float shadowFilter(float3 o, float3 d, float tmin, float tmax)
{
    if ((fd.flags.x & F_TRANSPARENT) == 0u)
        return occluded(o, d, tmin, tmax) ? 0.0 : 1.0;
    float f = 1.0;
    for (uint k = 0; k < 16u; k++)
    {
        Hit h = closestHit(o, d, tmin, tmax);
        if (!h.valid)
            break;
        float opacity = styleTable[2u * primStyle(h)].w;
        if (opacity >= 0.99999)
            return 0.0;
        f *= (1.0 - opacity);
        tmin = h.t + 1e-5 * max(1.0, h.t);
    }
    return f;
}

// Shading ///////////////////////////////////////////////////////////////////

struct Surface
{
    float3 p; // hit point
    float3 n; // normal facing the viewer
    float3 color;
    uint style;
};

float3 vec3At(StructuredBuffer<float> buf, uint i)
{
    return float3(buf[i], buf[i + 1], buf[i + 2]);
}

Surface fetchSurface(Hit h, float3 o, float3 d)
{
    Surface s;
    s.p = o + h.t * d;
    s.n = float3(0, 0, 1);
    s.color = float3(0, 0, 0);
    s.style = 0u;
    bool ownColor = false;
    InstInfo info = loadInst(h.inst);
    bool ident = (info.flags & 1u) != 0u;
    uint gi = info.base + h.prim;
    float3 oo = o;
    float3 od = d;
    if (!ident)
        toObjectRay(info, o, d, oo, od);

    if (info.type == 0u)
    {
        float4 sp = spheres[gi];
        float3 n = (oo + h.t * od - sp.xyz) / sp.w;
        s.n = ident ? n : normalize(rotate(info, n));
        ownColor = (fd.flags.x & F_SPHERE_COLORS) != 0u;
        if (ownColor)
            s.color = vec3At(sphereColors, 3u * gi);
        s.style = (fd.flags.x & F_SPHERE_STYLES) != 0u ? sphereStyles[gi] : fd.styles.x;
    }
    else if (info.type == 1u)
    {
        float t;
        float3 n;
        float sel;
        float dl = length(od);
        hitCylinder(gi, oo, od / dl, 0.0, (h.t * 1.0001 + 1e-6) * dl, t, n, sel);
        s.n = ident ? n : normalize(rotate(info, n));
        ownColor = (fd.flags.x & F_CYL_COLORS) != 0u;
        if (ownColor)
            s.color = sel < 0.5 ? vec3At(cylColors, 6u * gi)
                                : vec3At(cylColors, 6u * gi + 3u);
        s.style = (fd.flags.x & F_CYL_STYLES) != 0u ? cylStyles[gi] : fd.styles.y;
    }
    else
    {
        uint tb = 9u * gi;
        float3 v0 = vec3At(triPos, tb);
        float3 v1 = vec3At(triPos, tb + 3u);
        float3 v2 = vec3At(triPos, tb + 6u);
        uint sb = 12u * gi;
        float3 n0 = vec3At(triShade, sb);
        float3 n1 = vec3At(triShade, sb + 3u);
        float3 n2 = vec3At(triShade, sb + 6u);
        s.style = asuint(triShade[sb + 9u]);
        uint flags = asuint(triShade[sb + 10u]);
        uint aux = asuint(triShade[sb + 11u]);
        float w1 = h.bary.x;
        float w2 = h.bary.y;
        float w0 = 1.0 - w1 - w2;
        float3 nn = normalize(w0 * n0 + w1 * n1 + w2 * n2);
        float3 ng;
        if (ident)
            ng = cross(v1 - v0, v2 - v0);
        else
        {
            // like the CPU backend, orient by the winding of the world
            // space triangle; a mirroring transform reverses the winding
            ng = cross(toWorld(info, v1) - toWorld(info, v0),
                       toWorld(info, v2) - toWorld(info, v0));
            nn = normalize(rotate(info, nn));
            if ((info.flags & 2u) != 0u)
                nn = -nn;
        }
        s.n = dot(ng, d) > 0.0 ? -nn : nn;
        ownColor = (flags & 3u) != 0u;
        if ((flags & 1u) != 0u)
        {
            s.color = w0 * vec3At(triAux, aux) + w1 * vec3At(triAux, aux + 3u)
                      + w2 * vec3At(triAux, aux + 6u);
        }
        else if ((flags & 2u) != 0u)
        {
            float4 a0 = float4(triAux[aux], triAux[aux + 1], triAux[aux + 2], triAux[aux + 3]);
            float4 a1 = float4(triAux[aux + 4], triAux[aux + 5], triAux[aux + 6], triAux[aux + 7]);
            float4 a2 = float4(triAux[aux + 8], triAux[aux + 9], triAux[aux + 10], triAux[aux + 11]);
            float4 a = w0 * a0 + w1 * a1 + w2 * a2;
            s.color = evalSampler(flags >> 8, a).xyz;
        }
    }
    if (!ownColor)
        s.color = styleTable[2u * s.style].xyz;
    if (dot(s.n, d) > 0.0)
        s.n = -s.n;
    return s;
}

float3 shadeSurface(Hit h, Surface s, float3 d, out float alpha)
{
    float4 st0 = styleTable[2u * s.style];
    float4 st1 = styleTable[2u * s.style + 1u];
    alpha = st0.w;
    float3 col = s.color;
    float diffuseW = st1.x * fd.shade.y;
    float phong = st1.y;
    float phongExp = st1.z;
    bool metal = st1.w > 0.5;

    float3 diffuse = float3(0, 0, 0);
    float3 phongcol = float3(0, 0, 0);
    if (diffuseW > MINCONTRIB || phong > MINCONTRIB)
    {
        float biasScale = 1e-5 * max(1.0, max(abs(s.p.x), max(abs(s.p.y), abs(s.p.z))));
        float3 shadowO = s.p + s.n * biasScale;
        for (uint i = 0; i < fd.counts.w; i++)
        {
            float4 l0 = lights[2u * i];
            float3 lcol = lights[2u * i + 1u].xyz;
            float3 L;
            float Llen = 1e30;
            float atten = 1.0;
            if (l0.w < 0.5)
            {
                L = l0.xyz;
            }
            else
            {
                float3 lv = l0.xyz - s.p;
                float len = sqrt(dot(lv, lv)) + 1e-6;
                L = lv / len;
                Llen = len;
                atten = 1.0 / (Llen * Llen);
            }
            float inten = dot(s.n, L) * atten;
            if (inten > MINCONTRIB)
            {
                if ((fd.flags.x & F_SHADOWS) != 0u)
                {
                    float tmax = Llen >= 1e29 ? 1e30 : Llen - biasScale;
                    float filt = shadowFilter(shadowO, L, 0.0, tmax);
                    if (filt <= 0.0)
                        continue;
                    inten *= filt;
                }
                diffuse += lcol * inten;
                if (phong > MINCONTRIB)
                {
                    float3 H = L - d;
                    float ndh = dot(s.n, H);
                    float pv = 0.0;
                    if (ndh > MINCONTRIB)
                    {
                        ndh /= sqrt(dot(H, H));
                        pv = pow(ndh, phongExp);
                    }
                    phongcol += (metal ? col : lcol) * (pv * phong);
                }
            }
        }
    }
    diffuse *= diffuseW;
    float3 outc = col * (diffuse + fd.shade.x);
    if (phong > MINCONTRIB)
        outc += phongcol;
    return outc;
}

float3 radiance(float3 o, float3 d, bool recordDepth, out float depthOut)
{
    float3 result = float3(0, 0, 0);
    float w = 1.0;
    depthOut = 1e18;
    uint depthLeft = fd.flags.z;
    float tmin = 0.0;
    for (uint k = 0; k < 8u; k++)
    {
        Hit h = closestHit(o, d, tmin, 1e30);
        if (!h.valid)
        {
            result += w * fd.bg.xyz;
            break;
        }
        if (k == 0u)
            depthOut = h.t;
        Surface s = fetchSurface(h, o, d);
        float alpha;
        float3 c = shadeSurface(h, s, d, alpha);
        result += w * c;
        if (alpha < 1.0 - MINCONTRIB)
        {
            if (depthLeft <= 1u)
            {
                result += w * fd.bg.xyz;
                break;
            }
            w *= (1.0 - alpha);
            depthLeft--;
            float eps = 1e-5 * max(1.0, max(abs(s.p.x), max(abs(s.p.y), abs(s.p.z))));
            o = s.p;
            tmin = eps;
        }
        else
            break;
    }
    return result;
}

// Frame ////////////////////////////////////////////////////////////////////

uint pcg(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float2 jitter(uint pixel, uint sampleIndex)
{
    uint h = pcg(pixel ^ pcg(sampleIndex * 0x9E3779B9u + 0x7F4A7C15u));
    uint h2 = pcg(h + 0x68E31DA4u);
    return float2(float(h >> 8) * (1.0 / 16777216.0) - 0.5,
                  float(h2 >> 8) * (1.0 / 16777216.0) - 0.5);
}

float toSrgb8(float v)
{
    v = clamp(v, 0.0, 1.0);
    v = pow(v, 1.0 / 2.2);
    return float(uint(255.0 * clamp(v, 0.0, 1.0)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint W = fd.size.x;
    uint H = fd.size.y;
    if (tid.x >= W || tid.y >= H)
        return;
    uint pixel = tid.y * W + tid.x;
    float3 fwd = fd.fwdU0.xyz;

    float3 sum = float3(0, 0, 0);
    float depth0 = 1e18;
    for (uint s = 0; s < fd.size.z; s++)
    {
        uint g = fd.size.w + s; // global sample index
        float2 j = g == 0u ? float2(0, 0) : jitter(pixel, g);
        float u = fd.fwdU0.w + fd.rightDu.w * (float(tid.x) + 0.5 + j.x);
        float v = fd.upV0.w + fd.shade.z * (float(tid.y) + 0.5 + j.y);
        float3 plane = fd.rightDu.xyz * u + fd.upV0.xyz * v;
        float3 o;
        float3 d;
        if (fd.posOrtho.w > 0.5)
        {
            o = fd.posOrtho.xyz + plane;
            d = fwd;
        }
        else
        {
            o = fd.posOrtho.xyz;
            d = normalize(fwd + plane);
        }
        float dep;
        sum += radiance(o, d, g == 0u, dep);
        if (g == 0u)
            depth0 = dep;
    }

    uint total = fd.size.w + fd.size.z; // samples in the average
    if ((fd.flags.x & F_ACCUM) != 0u)
    {
        if ((fd.flags.x & F_ACCUM_READ) != 0u)
            sum += float3(accum[3 * pixel], accum[3 * pixel + 1], accum[3 * pixel + 2]);
        accum[3 * pixel + 0] = sum.x;
        accum[3 * pixel + 1] = sum.y;
        accum[3 * pixel + 2] = sum.z;
    }
    float3 rgb = sum / float(total);

    // Output in the frame's pixel format, ANARI row order (bottom row first)
    uint row = H - 1u - tid.y;
    uint dst = row * W + tid.x;
    float alpha = 1.0;
    if ((fd.flags.x & F_BG_ALPHA) != 0u && all(rgb == fd.bg.xyz))
        alpha = fd.bg.w;
    uint fmt = fd.flags.y;
    if (fmt == FMT_F32)
    {
        float3 c = min(rgb, float3(1, 1, 1));
        outPixels[4 * dst + 0] = asuint(c.x);
        outPixels[4 * dst + 1] = asuint(c.y);
        outPixels[4 * dst + 2] = asuint(c.z);
        outPixels[4 * dst + 3] = asuint(alpha);
    }
    else if (fmt == FMT_U8 || fmt == FMT_U8_SRGB)
    {
        float3 c = min(rgb, float3(1, 1, 1));
        uint r, g, b;
        if (fmt == FMT_U8)
        {
            r = uint(255.0 * clamp(c.x, 0.0, 1.0));
            g = uint(255.0 * clamp(c.y, 0.0, 1.0));
            b = uint(255.0 * clamp(c.z, 0.0, 1.0));
        }
        else
        {
            r = uint(toSrgb8(c.x));
            g = uint(toSrgb8(c.y));
            b = uint(toSrgb8(c.z));
        }
        uint a = uint(255.0 * clamp(alpha, 0.0, 1.0));
        outPixels[dst] = r | (g << 8) | (b << 16) | (a << 24);
    }
    if ((fd.flags.x & F_DEPTH) != 0u && fd.size.w == 0u)
        outDepth[dst] = depth0 >= 1e17 ? asfloat(0x7F800000u) : depth0;
}
