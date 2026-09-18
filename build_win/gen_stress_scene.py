#!/usr/bin/env python3
"""gen_stress_scene.py - generate a synthetic "clusters in a stadium" scene:
many small dense clusters of spheres scattered across a huge empty volume,
with the camera looking at one cluster close up. This is the classic case
where a uniform grid (sized to the whole scene's bounding box) performs
badly -- cells are huge relative to the geometry, so a lot of near-empty
cells get stepped through and/or overloaded cells end up with far too many
objects -- while a BVH's per-node bounds stay tight around actual geometry
regardless of how sparse the surrounding volume is. Tachyon's own benchmark
scenes (37-2330 objects, tens of units across) are all small/compact enough
that this gap barely shows; this scene is built specifically to surface it.

Usage: python3 gen_stress_scene.py > scenes/stress_clusters.dat
"""
import random

random.seed(42)

NUM_CLUSTERS = 24
SPHERES_PER_CLUSTER = 2500
CLUSTER_SPREAD = 4000.0   # clusters scattered across +/- this range
CLUSTER_RADIUS = 6.0      # spheres within a cluster packed into this radius
SPHERE_RAD = 0.12

print("BEGIN_SCENE")
print("  RESOLUTION 800 600")
print()
print("CAMERA")
print("  ZOOM  1.0")
print("  ASPECTRATIO 1.0")
print("  ANTIALIASING 0")
print("  RAYDEPTH  4")
print("  CENTER    0.0 0.0 -18.0")
print("  VIEWDIR   0.0 0.0 1.0")
print("  UPDIR     0.0 1.0 0.0")
print("END_CAMERA")
print()
print("LIGHT")
print("  CENTER -10.0 10.0 -15.0")
print("  RAD     1.0")
print("  COLOR   1.0 1.0 1.0")
print()
print("LIGHT")
print("  CENTER  10.0 5.0 -10.0")
print("  RAD     1.0")
print("  COLOR   0.6 0.6 0.6")
print()

# Cluster 0 is centered at the origin, in front of the camera, so the
# rendered image is a real close-up view, not an empty frame.
centers = [(0.0, 0.0, 0.0)]
for _ in range(NUM_CLUSTERS - 1):
    centers.append((
        random.uniform(-CLUSTER_SPREAD, CLUSTER_SPREAD),
        random.uniform(-CLUSTER_SPREAD, CLUSTER_SPREAD),
        random.uniform(-CLUSTER_SPREAD, CLUSTER_SPREAD),
    ))

for cx, cy, cz in centers:
    for _ in range(SPHERES_PER_CLUSTER):
        x = cx + random.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
        y = cy + random.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
        z = cz + random.uniform(-CLUSTER_RADIUS, CLUSTER_RADIUS)
        r = random.random()
        g = random.random()
        b = random.random()
        print("SPHERE")
        print(f"  CENTER {x:.4f} {y:.4f} {z:.4f}")
        print(f"  RAD {SPHERE_RAD}")
        print("  TEXTURE")
        print(f"    AMBIENT 0.1 DIFFUSE 0.8 SPECULAR 0.1 OPACITY 1.0 COLOR {r:.3f} {g:.3f} {b:.3f} TEXFUNC 0")

print()
print("END_SCENE")
