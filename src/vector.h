/*
 * vector.h - This file contains declarations of vector functions
 *
 *  $Id: vector.h,v 1.5 2002/07/09 00:48:40 johns Exp $
 */

#ifndef RT_VECTOR_H
#define RT_VECTOR_H 1

/*
 * These are tiny, allocation-free leaf functions called tens of millions
 * of times per render (VDot/VCross alone were 25-40M calls in a single
 * benchmark frame). Previously they were ordinary externally-linked
 * functions in vector.c, so every call paid for a real cross-TU function
 * call even though the body is 1-3 FP ops. static inline here lets every
 * caller fold them in directly -- same arithmetic, same operation order,
 * just no call/return overhead. (Every file that includes this header
 * already #defines TACHYON_INTERNAL before #include "tachyon.h", so the
 * vector/color/ray struct layouts this needs are visible everywhere it's
 * used; see tachyon.h's TACHYON_INTERNAL-gated section.)
 */

static inline flt VDot(const vector *a, const vector *b) {
  return (a->x*b->x + a->y*b->y + a->z*b->z);
}

static inline void VCross(const vector * a, const vector * b, vector * c) {
  c->x = (a->y * b->z) - (a->z * b->y);
  c->y = (a->z * b->x) - (a->x * b->z);
  c->z = (a->x * b->y) - (a->y * b->x);
}

static inline flt VLength(const vector * a) {
  return (flt) SQRT((a->x * a->x) + (a->y * a->y) + (a->z * a->z));
}

static inline void VNorm(vector * a) {
  flt len;

  len=SQRT((a->x * a->x) + (a->y * a->y) + (a->z * a->z));
  if (len != 0.0) {
    a->x /= len;
    a->y /= len;
    a->z /= len;
  }
}

static inline void VAdd(const vector * a, const vector * b, vector * c) {
  c->x = (a->x + b->x);
  c->y = (a->y + b->y);
  c->z = (a->z + b->z);
}

static inline void VSub(const vector * a, const vector * b, vector * c) {
  c->x = (a->x - b->x);
  c->y = (a->y - b->y);
  c->z = (a->z - b->z);
}

static inline void VAddS(flt a, const vector * A, const vector * B, vector * C) {
  C->x = (a * A->x) + B->x;
  C->y = (a * A->y) + B->y;
  C->z = (a * A->z) + B->z;
}

static inline vector Raypnt(const ray * a, flt t) {
  vector temp;

  temp.x=a->o.x + (a->d.x * t);
  temp.y=a->o.y + (a->d.y * t);
  temp.z=a->o.z + (a->d.z * t);

  return temp;
}

static inline void VScale(vector * a, flt s) {
  a->x *= s;
  a->y *= s;
  a->z *= s;
}

static inline void ColorAddS(color * a, const color * b, flt s) {
  a->r += b->r * s;
  a->g += b->g * s;
  a->b += b->b * s;
}

static inline void ColorAccum(color * a, const color * b) {
  a->r += b->r;
  a->g += b->g;
  a->b += b->b;
}

static inline void ColorScale(color * a, flt s) {
  a->r *= s;
  a->g *= s;
  a->b *= s;
}

#endif
