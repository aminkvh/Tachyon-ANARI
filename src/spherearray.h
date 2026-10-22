/*
 * spherearray.h - a single bounded object holding many spheres (SoA float
 *                 arrays) with its own compact bounding volume hierarchy.
 *
 * Meant for scenes with very many spheres (e.g. molecules, particles): where
 * one object per sphere costs a few hundred bytes each plus build time in the
 * scene-wide acceleration structure, this costs about 40 bytes per sphere
 * and builds a Morton-order (LBVH) tree in a few tens of milliseconds per
 * million spheres.
 *
 * The primitive that was hit is carried in ray->intstruct.prim, which the
 * array's normal and (optional) per-sphere color texture functions read.
 */

/* Takes ownership of nothing: all input is copied. 'rad' (n floats) may be
 * NULL, then 'uniformrad' is used. 'rgb' (3n floats) may be NULL, then the
 * sphere color comes from 'tex'; otherwise a per-array copy of 'tex' is used
 * that picks the color by sphere. 'outtex' receives the texture the object
 * actually uses (== tex, or the per-array copy, which the caller may need
 * for shading parameter updates); may be NULL. */
object * newspherearray(void * tex, int n, const float * ctr,
                        const float * rad, float uniformrad,
                        const float * rgb, void ** outtex);
