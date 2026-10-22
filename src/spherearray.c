/*
 * spherearray.c - many spheres in one object, see spherearray.h.
 *
 * Layout: the spheres are stored in Morton (Z-curve) order as separate
 * float arrays; a binary BVH over them is built by splitting the sorted
 * order at the highest differing Morton bit (LBVH). Only interior nodes
 * exist as nodes: each holds the boxes of both of its children (so both are
 * tested together with a few SSE operations) and a reference to each child,
 * which is either another interior node or a run of at most SA_MAX_LEAF
 * spheres.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define TACHYON_INTERNAL 1
#include "tachyon.h"
#include "macros.h"
#include "vector.h"
#include "intersect.h"
#include "util.h"
#include "spherearray.h"

#if defined(__SSE__) || defined(_M_X64) || defined(__x86_64__)
#define SA_USE_SSE 1
#include <xmmintrin.h>
#endif

#define SA_MAX_LEAF   4
#define SA_STACK_SIZE 96

/* child reference: bit 31 set => a leaf: bits 28-30 = sphere count - 1,
 * bits 0-27 = index of the first sphere; otherwise an interior node index */
#define SA_LEAF_FLAG   0x80000000u
#define SA_MAX_SPHERES 0x0FFFFFFFu

/* 64-byte interior node. cbox holds the boxes of the two children as
 * structure-of-arrays pairs: minx[2] miny[2] minz[2] maxx[2] maxy[2] maxz[2] */
typedef struct {
  float cbox[12];
  unsigned int child[2];
  unsigned int pad[2];
} sanode;

typedef struct {
  RT_OBJECT_HEAD
  int n;
  int numnodes;
  float * ctr;           /**< 3n, in BVH order */
  float * rad;           /**< n, in BVH order, or NULL when uniform */
  float uniformrad;
  float * rgb;           /**< 3n, in BVH order, or NULL */
  sanode * nodes;
  unsigned int root;     /**< child reference of the root */
  float rootbox[6];      /**< bounds: min xyz, max xyz */
  standard_texture ctex; /**< per-array texture copy, only used with rgb */
} spherearray;

static int spherearray_bbox(void * obj, vector * min, vector * max);
static void spherearray_free(void * obj);
static void spherearray_intersect(const spherearray *, ray *);
static void spherearray_normal(const spherearray *, const vector *, const ray *, vector *);

static object_methods spherearray_methods = {
  (void (*)(const void *, void *))(spherearray_intersect),
  (void (*)(const void *, const void *, const void *, void *))(spherearray_normal),
  spherearray_bbox,
  spherearray_free
};

static void sa_noop_freetex(void * p) {
  (void) p;
}

static texture_methods sa_tex_methods = {
  sa_noop_freetex
};

static void spherearray_free(void * obj) {
  spherearray * a = (spherearray *) obj;
  free(a->ctr);
  free(a->rad);
  free(a->rgb);
  free(a->nodes);
  free(a);
}

static int spherearray_bbox(void * obj, vector * min, vector * max) {
  spherearray * a = (spherearray *) obj;
  min->x = a->rootbox[0];
  min->y = a->rootbox[1];
  min->z = a->rootbox[2];
  max->x = a->rootbox[3];
  max->y = a->rootbox[4];
  max->z = a->rootbox[5];
  return 1;
}

/* per-sphere color: the sphere that was hit is in ray->intstruct.prim */
static color spherearray_texfunc(const void * hit, const void * tx, void * voidray) {
  const ray * ry = (const ray *) voidray;
  const spherearray * a = (const spherearray *) ry->intstruct.closest.obj;
  const float * c = &a->rgb[3 * (size_t) ry->intstruct.prim];
  color col;
  (void) hit; (void) tx;
  col.r = c[0];
  col.g = c[1];
  col.b = c[2];
  return col;
}

/*
 * ---- Build: Morton-order LBVH ----
 */

typedef struct {
  const float * ctr;      /* original (unsorted) centers */
  const float * rad;
  float uniformrad;
  unsigned int * keys;    /* sorted Morton codes */
  unsigned int * perm;    /* sorted -> original index */
  sanode * nodes;
  int numnodes;
} sabuild;

static unsigned int sa_expand10(unsigned int v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

/* Builds the subtree over sorted range [lo,hi): returns its child reference
 * and its box (min xyz, max xyz). */
static unsigned int sa_build_range(sabuild * b, int lo, int hi, float * box) {
  int i, k;

  if ((hi - lo) <= SA_MAX_LEAF) {
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    for (i = lo; i < hi; i++) {
      unsigned int o = b->perm[i];
      float r = b->rad ? b->rad[o] : b->uniformrad;
      for (k = 0; k < 3; k++) {
        float c = b->ctr[3 * (size_t) o + k];
        if (c - r < mn[k]) mn[k] = c - r;
        if (c + r > mx[k]) mx[k] = c + r;
      }
    }
    /* widen by a few ulps so rounding can never shrink the box below the
     * true sphere extent */
    for (k = 0; k < 3; k++) {
      box[k]     = mn[k] - fabsf(mn[k]) * 4.8e-7f - 1e-30f;
      box[3 + k] = mx[k] + fabsf(mx[k]) * 4.8e-7f + 1e-30f;
    }
    return SA_LEAF_FLAG | ((unsigned int) (hi - lo - 1) << 28) | (unsigned int) lo;
  } else {
    unsigned int c0 = b->keys[lo], c1 = b->keys[hi - 1];
    int mid, idx;
    unsigned int lref, rref;
    float lbox[6], rbox[6];
    sanode * node;

    if (c0 == c1) {
      mid = (lo + hi) / 2;
    } else {
      /* first element whose bit 'bit' is set (all share the higher prefix) */
      unsigned int diff = c0 ^ c1;
      int bit = 31;
      int l = lo, h = hi - 1;
      while (!((diff >> bit) & 1u)) bit--;
      while (l < h) {
        int m = (l + h) / 2;
        if ((b->keys[m] >> bit) & 1u) h = m; else l = m + 1;
      }
      mid = l;
    }

    idx = b->numnodes++;
    lref = sa_build_range(b, lo, mid, lbox);
    rref = sa_build_range(b, mid, hi, rbox);
    node = &b->nodes[idx];
    node->child[0] = lref;
    node->child[1] = rref;
    node->pad[0] = node->pad[1] = 0;
    for (k = 0; k < 3; k++) {
      node->cbox[2 * k + 0] = lbox[k];         /* min */
      node->cbox[2 * k + 1] = rbox[k];
      node->cbox[6 + 2 * k + 0] = lbox[3 + k]; /* max */
      node->cbox[6 + 2 * k + 1] = rbox[3 + k];
      box[k]     = lbox[k] < rbox[k] ? lbox[k] : rbox[k];
      box[3 + k] = lbox[3 + k] > rbox[3 + k] ? lbox[3 + k] : rbox[3 + k];
    }
    return (unsigned int) idx;
  }
}

static void sa_radix_sort(unsigned int * keys, unsigned int * perm, int n) {
  unsigned int * k2 = (unsigned int *) malloc(sizeof(unsigned int) * n);
  unsigned int * p2 = (unsigned int *) malloc(sizeof(unsigned int) * n);
  unsigned int * ka = keys, * pa = perm, * kb = k2, * pb = p2, * t;
  int pass, i;

  for (pass = 0; pass < 3; pass++) {
    unsigned int count[1025];
    int shift = pass * 10;
    memset(count, 0, sizeof(count));
    for (i = 0; i < n; i++)
      count[((ka[i] >> shift) & 1023u) + 1]++;
    for (i = 0; i < 1024; i++)
      count[i + 1] += count[i];
    for (i = 0; i < n; i++) {
      unsigned int d = count[(ka[i] >> shift) & 1023u]++;
      kb[d] = ka[i];
      pb[d] = pa[i];
    }
    t = ka; ka = kb; kb = t;
    t = pa; pa = pb; pb = t;
  }
  /* three passes: the sorted result ended up in k2/p2 */
  memcpy(keys, ka, sizeof(unsigned int) * n);
  memcpy(perm, pa, sizeof(unsigned int) * n);
  free(k2);
  free(p2);
}

object * newspherearray(void * tex, int n, const float * ctr,
                        const float * rad, float uniformrad,
                        const float * rgb, void ** outtex) {
  spherearray * a;
  sabuild b;
  float cmin[3], cmax[3], scale[3];
  int i, k;

  if (outtex != NULL)
    *outtex = tex;
  if (n <= 0 || (unsigned int) n > SA_MAX_SPHERES)
    return NULL;

  a = (spherearray *) malloc(sizeof(spherearray));
  memset(a, 0, sizeof(spherearray));
  a->methods = &spherearray_methods;
  a->n = n;
  a->uniformrad = uniformrad;
  a->tex = (texture *) tex;

  b.keys = (unsigned int *) malloc(sizeof(unsigned int) * n);
  b.perm = (unsigned int *) malloc(sizeof(unsigned int) * n);

  for (k = 0; k < 3; k++) {
    cmin[k] = 1e30f;
    cmax[k] = -1e30f;
  }
  for (i = 0; i < n; i++) {
    for (k = 0; k < 3; k++) {
      float c = ctr[3 * (size_t) i + k];
      if (c < cmin[k]) cmin[k] = c;
      if (c > cmax[k]) cmax[k] = c;
    }
  }
  for (k = 0; k < 3; k++) {
    float e = cmax[k] - cmin[k];
    scale[k] = e > 0.0f ? 1023.0f / e : 0.0f;
  }
  for (i = 0; i < n; i++) {
    unsigned int q[3];
    for (k = 0; k < 3; k++) {
      float f = (ctr[3 * (size_t) i + k] - cmin[k]) * scale[k];
      q[k] = f <= 0.0f ? 0u : (f >= 1023.0f ? 1023u : (unsigned int) f);
    }
    b.keys[i] = (sa_expand10(q[0]) << 2) | (sa_expand10(q[1]) << 1) | sa_expand10(q[2]);
    b.perm[i] = (unsigned int) i;
  }
  sa_radix_sort(b.keys, b.perm, n);

  b.ctr = ctr;
  b.rad = rad;
  b.uniformrad = uniformrad;
  /* interior nodes <= leaves - 1 <= n - 1; untouched pages cost nothing and
   * the array is shrunk below */
  b.nodes = (sanode *) malloc(sizeof(sanode) * (size_t) n);
  b.numnodes = 0;
  a->root = sa_build_range(&b, 0, n, a->rootbox);
  a->numnodes = b.numnodes;
  if (b.numnodes > 0) {
    a->nodes = (sanode *) realloc(b.nodes, sizeof(sanode) * (size_t) b.numnodes);
  } else {
    free(b.nodes);
    a->nodes = NULL;
  }

  /* gather the sphere data into BVH order */
  a->ctr = (float *) malloc(sizeof(float) * 3 * (size_t) n);
  for (i = 0; i < n; i++) {
    const float * s = &ctr[3 * (size_t) b.perm[i]];
    a->ctr[3 * i + 0] = s[0];
    a->ctr[3 * i + 1] = s[1];
    a->ctr[3 * i + 2] = s[2];
  }
  if (rad != NULL) {
    a->rad = (float *) malloc(sizeof(float) * (size_t) n);
    for (i = 0; i < n; i++)
      a->rad[i] = rad[b.perm[i]];
  }
  if (rgb != NULL) {
    a->rgb = (float *) malloc(sizeof(float) * 3 * (size_t) n);
    for (i = 0; i < n; i++) {
      const float * s = &rgb[3 * (size_t) b.perm[i]];
      a->rgb[3 * i + 0] = s[0];
      a->rgb[3 * i + 1] = s[1];
      a->rgb[3 * i + 2] = s[2];
    }
    memcpy(&a->ctex, tex, sizeof(standard_texture));
    a->ctex.texfunc = (color(*)(const void *, const void *, void *))(spherearray_texfunc);
    a->ctex.methods = &sa_tex_methods;
    a->tex = (texture *) &a->ctex;
    if (outtex != NULL)
      *outtex = &a->ctex;
  }

  free(b.keys);
  free(b.perm);
  return (object *) a;
}

/*
 * ---- Traversal ----
 */

/* Offers a sphere hit to the ray, remembering which sphere it was. Same
 * acceptance logic as individual spheres: add_intersection() decides, and
 * 'prim' is only recorded when the hit became the closest one. */
static void sa_offer(const spherearray * a, ray * ry, flt t, int idx) {
  if (t > EPSILON && t < ry->maxdist) {
    ry->add_intersection(t, (const object *) a, ry);
    if (ry->intstruct.closest.obj == (const object *) a && ry->intstruct.closest.t == t)
      ry->intstruct.prim = (unsigned int) idx;
  }
}

static void sa_intersect_sphere(const spherearray * a, int i, ray * ry) {
  flt b, disc, t1, t2, temp, rad;
  vector V;

  /* same arithmetic as sphere_intersect() */
  V.x = (flt) a->ctr[3 * (size_t) i + 0] - ry->o.x;
  V.y = (flt) a->ctr[3 * (size_t) i + 1] - ry->o.y;
  V.z = (flt) a->ctr[3 * (size_t) i + 2] - ry->o.z;
  rad = (flt) (a->rad ? a->rad[i] : a->uniformrad);
  VDOT(b, V, ry->d);
  VDOT(temp, V, V);

  disc=b*b + rad*rad - temp;
  if (disc<=0.0) return;
  disc=SQRT(disc);

  t2=b+disc;
  if (t2 <= SPEPSILON)
    return;
  sa_offer(a, ry, t2, i);

  t1=b-disc;
  if (t1 > SPEPSILON)
    sa_offer(a, ry, t1, i);
}

typedef struct {
  unsigned int ref;
  float tmin;
} sa_stackentry;

/* Slab test of one box (min xyz at b[0..2], max xyz at b[3..5]); the extra
 * factor on tmax keeps single-precision rounding from rejecting grazing
 * hits. */
static int sa_box_hit(const float * b, const float * o, const float * invd,
                      float maxdist, float * outtmin) {
  float tx1 = (b[0] - o[0]) * invd[0], tx2 = (b[3] - o[0]) * invd[0];
  float ty1 = (b[1] - o[1]) * invd[1], ty2 = (b[4] - o[1]) * invd[1];
  float tz1 = (b[2] - o[2]) * invd[2], tz2 = (b[5] - o[2]) * invd[2];
  float tmin = MYMAX(MYMAX(MYMIN(tx1, tx2), MYMIN(ty1, ty2)), MYMAX(MYMIN(tz1, tz2), 0.0f));
  float tmax = MYMIN(MYMIN(MYMAX(tx1, tx2), MYMAX(ty1, ty2)), MYMAX(tz1, tz2)) * 1.000004f;
  *outtmin = tmin;
  return (tmin <= tmax) && (tmin <= maxdist);
}

/* Tests both children of a node: returns a bit mask (bit 0 = child 0) and
 * the entry distances. */
static int sa_pair_hit(const sanode * n, const float * o, const float * invd,
                       float maxdist, float * tmin) {
#if defined(SA_USE_SSE)
  const __m128 zero = _mm_setzero_ps();
  const __m128 ox = _mm_set1_ps(o[0]), oy = _mm_set1_ps(o[1]), oz = _mm_set1_ps(o[2]);
  const __m128 ix = _mm_set1_ps(invd[0]), iy = _mm_set1_ps(invd[1]), iz = _mm_set1_ps(invd[2]);
  __m128 minx = _mm_loadl_pi(zero, (const __m64 *) &n->cbox[0]);
  __m128 miny = _mm_loadl_pi(zero, (const __m64 *) &n->cbox[2]);
  __m128 minz = _mm_loadl_pi(zero, (const __m64 *) &n->cbox[4]);
  __m128 maxx = _mm_loadl_pi(zero, (const __m64 *) &n->cbox[6]);
  __m128 maxy = _mm_loadl_pi(zero, (const __m64 *) &n->cbox[8]);
  __m128 maxz = _mm_loadl_pi(zero, (const __m64 *) &n->cbox[10]);
  __m128 tx1 = _mm_mul_ps(_mm_sub_ps(minx, ox), ix);
  __m128 tx2 = _mm_mul_ps(_mm_sub_ps(maxx, ox), ix);
  __m128 ty1 = _mm_mul_ps(_mm_sub_ps(miny, oy), iy);
  __m128 ty2 = _mm_mul_ps(_mm_sub_ps(maxy, oy), iy);
  __m128 tz1 = _mm_mul_ps(_mm_sub_ps(minz, oz), iz);
  __m128 tz2 = _mm_mul_ps(_mm_sub_ps(maxz, oz), iz);
  __m128 tn = _mm_max_ps(_mm_max_ps(_mm_min_ps(tx1, tx2), _mm_min_ps(ty1, ty2)),
                         _mm_max_ps(_mm_min_ps(tz1, tz2), zero));
  __m128 tf = _mm_mul_ps(_mm_min_ps(_mm_min_ps(_mm_max_ps(tx1, tx2), _mm_max_ps(ty1, ty2)),
                                    _mm_max_ps(tz1, tz2)), _mm_set1_ps(1.000004f));
  __m128 hit = _mm_and_ps(_mm_cmple_ps(tn, tf), _mm_cmple_ps(tn, _mm_set1_ps(maxdist)));
  float tmp[4];
  _mm_storeu_ps(tmp, tn);
  tmin[0] = tmp[0];
  tmin[1] = tmp[1];
  return _mm_movemask_ps(hit) & 3;
#else
  float b0[6], b1[6];
  int k, mask = 0;
  for (k = 0; k < 3; k++) {
    b0[k] = n->cbox[2 * k];         b1[k] = n->cbox[2 * k + 1];
    b0[3 + k] = n->cbox[6 + 2 * k]; b1[3 + k] = n->cbox[6 + 2 * k + 1];
  }
  if (sa_box_hit(b0, o, invd, maxdist, &tmin[0])) mask |= 1;
  if (sa_box_hit(b1, o, invd, maxdist, &tmin[1])) mask |= 2;
  return mask;
#endif
}

static void sa_intersect_leaf(const spherearray * a, unsigned int ref, ray * ry) {
  int first = (int) (ref & 0x0FFFFFFFu);
  int cnt = (int) ((ref >> 28) & 7u) + 1;
  int i;
  for (i = first; i < first + cnt; i++) {
    sa_intersect_sphere(a, i, ry);
    if (ry->flags & RT_RAY_FINISHED)
      return;
  }
}

static float sa_maxdist(const ray * ry) {
  return ry->maxdist > 3.0e38 ? 3.0e38f : (float) ry->maxdist;
}

static void spherearray_intersect(const spherearray * a, ray * ry) {
  sa_stackentry stack[SA_STACK_SIZE];
  int sp = 0;
  unsigned int ref = a->root;
  float o[3], invd[3], tmin;

  if (ry->flags & RT_RAY_FINISHED)
    return;

  o[0] = (float) ry->o.x; o[1] = (float) ry->o.y; o[2] = (float) ry->o.z;
  invd[0] = (float) (1.0 / ry->d.x);
  invd[1] = (float) (1.0 / ry->d.y);
  invd[2] = (float) (1.0 / ry->d.z);

  if (!sa_box_hit(a->rootbox, o, invd, sa_maxdist(ry), &tmin))
    return;

  for (;;) {
    if (ref & SA_LEAF_FLAG) {
      sa_intersect_leaf(a, ref, ry);
      if (ry->flags & RT_RAY_FINISHED)
        return;
    } else {
      const sanode * node = &a->nodes[ref];
      float tm[2];
      int mask = sa_pair_hit(node, o, invd, sa_maxdist(ry), tm);

      if (mask == 3) {
        int nearc = tm[1] < tm[0]; /* visit the closer child first */
        stack[sp].ref = node->child[1 - nearc];
        stack[sp].tmin = tm[1 - nearc];
        sp++;
        ref = node->child[nearc];
        continue;
      } else if (mask) {
        ref = node->child[mask - 1]; /* mask is 1 or 2 */
        continue;
      }
    }

    /* nothing more below here: pop the next still-live entry */
    for (;;) {
      if (sp == 0)
        return;
      sp--;
      if (stack[sp].tmin <= sa_maxdist(ry)) {
        ref = stack[sp].ref;
        break;
      }
    }
  }
}

static void spherearray_normal(const spherearray * a, const vector * pnt, const ray * incident, vector * N) {
  const float * c = &a->ctr[3 * (size_t) incident->intstruct.prim];
  flt invlen;

  N->x = pnt->x - c[0];
  N->y = pnt->y - c[1];
  N->z = pnt->z - c[2];

  invlen = 1.0 / SQRT(N->x*N->x + N->y*N->y + N->z*N->z);
  N->x *= invlen;
  N->y *= invlen;
  N->z *= invlen;

  /* Flip surface normal to point toward the viewer if necessary */
  if (VDot(N, &(incident->d)) > 0.0)  {
    N->x=-N->x;
    N->y=-N->y;
    N->z=-N->z;
  }
}
