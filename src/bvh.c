/*
 * bvh.c - bounding volume hierarchy acceleration structure, built with
 *         binned surface-area-heuristic (SAH) splits.
 *
 * Alternative to grid.c's uniform grid, same object_methods interface, so
 * render.c can use either one interchangeably as the thing it hangs on
 * scene->objgroup.boundedobj. See grid.c for the sibling implementation
 * and comments on the object list conventions (unboundedobj vs
 * boundedobj, the RT_OBJECT_HEAD/id/mailbox scheme, etc.) this mirrors.
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
#include "ui.h"

#define BVH_PRIVATE
#include "bvh.h"

static object_methods bvh_methods = {
  (void (*)(const void *, void *))(bvh_intersect),
  (void (*)(const void *, const void *, const void *, void *))(NULL),
  bvh_bbox,
  bvh_free
};

static int bvh_bbox(void * obj, vector * min, vector * max) {
  bvh * b = (bvh *) obj;

  if (b->numnodes == 0)
    return 0; /* empty BVH, shouldn't normally happen */

  *min = b->nodes[0].bmin;
  *max = b->nodes[0].bmax;
  return 1;
}

static void bvh_free(void * v) {
  bvh * b = (bvh *) v;

  free(b->nodes);
  free(b->primitives);
  free_objects(b->objects);
  free(b);
}


/*
 * ---- Build ----
 */

typedef struct {
  vector bmin;
  vector bmax;
  vector centroid;
  object * obj;
} bvh_buildprim;

typedef struct {
  bvh_buildprim * prims;
  bvhnode * nodes;
  int numnodes;
  int maxnodes;
} bvh_buildstate;

static flt vcomp(const vector * v, int axis) {
  return (axis == 0) ? v->x : (axis == 1) ? v->y : v->z;
}

static flt box_surface_area(const vector * bmin, const vector * bmax) {
  flt dx = bmax->x - bmin->x;
  flt dy = bmax->y - bmin->y;
  flt dz = bmax->z - bmin->z;
  if (dx < 0.0 || dy < 0.0 || dz < 0.0)
    return 0.0; /* degenerate/empty */
  return 2.0 * (dx*dy + dy*dz + dz*dx);
}

static void box_union(vector * bmin, vector * bmax,
                       const vector * omin, const vector * omax) {
  bmin->x = MYMIN(bmin->x, omin->x);
  bmin->y = MYMIN(bmin->y, omin->y);
  bmin->z = MYMIN(bmin->z, omin->z);
  bmax->x = MYMAX(bmax->x, omax->x);
  bmax->y = MYMAX(bmax->y, omax->y);
  bmax->z = MYMAX(bmax->z, omax->z);
}

/* Bounds of prims[start,end), both the primitive bounding boxes (bbox) and
 * their centroids (cbox) -- the caller needs both: bbox for the node's own
 * AABB, cbox to decide the split axis and bin boundaries. */
static void compute_bounds(const bvh_buildprim * prims, int start, int end,
                            vector * bmin, vector * bmax,
                            vector * cmin, vector * cmax) {
  int i;
  bmin->x = cmin->x =  FHUGE; bmin->y = cmin->y =  FHUGE; bmin->z = cmin->z =  FHUGE;
  bmax->x = cmax->x = -FHUGE; bmax->y = cmax->y = -FHUGE; bmax->z = cmax->z = -FHUGE;

  for (i = start; i < end; i++) {
    box_union(bmin, bmax, &prims[i].bmin, &prims[i].bmax);

    cmin->x = MYMIN(cmin->x, prims[i].centroid.x);
    cmin->y = MYMIN(cmin->y, prims[i].centroid.y);
    cmin->z = MYMIN(cmin->z, prims[i].centroid.z);
    cmax->x = MYMAX(cmax->x, prims[i].centroid.x);
    cmax->y = MYMAX(cmax->y, prims[i].centroid.y);
    cmax->z = MYMAX(cmax->z, prims[i].centroid.z);
  }
}

typedef struct {
  int count;
  vector bmin, bmax;
} bvh_bin;

static void bin_reset(bvh_bin * bins) {
  int i;
  for (i = 0; i < BVH_NUM_BINS; i++) {
    bins[i].count = 0;
    bins[i].bmin.x = bins[i].bmin.y = bins[i].bmin.z =  FHUGE;
    bins[i].bmax.x = bins[i].bmax.y = bins[i].bmax.z = -FHUGE;
  }
}

/* Which bin a centroid falls into along axis, given the centroid bounds
 * cmin[axis]..cmax[axis] (non-degenerate: caller already checked). */
static int bin_index(flt c, flt cminv, flt cmaxv) {
  int idx = (int) (BVH_NUM_BINS * (c - cminv) / (cmaxv - cminv));
  if (idx < 0) idx = 0;
  if (idx >= BVH_NUM_BINS) idx = BVH_NUM_BINS - 1;
  return idx;
}

/* Evaluate binned SAH for one axis; on a strictly better candidate, updates
 * bestcost, bestaxis and bestsplit (bestsplit = bin index: split is
 * between bin bestsplit and bestsplit+1). Returns nothing -- purely an
 * in/out comparison against the running best across all 3 axes. */
static void evaluate_axis_sah(const bvh_buildprim * prims, int start, int end,
                               int axis, flt cminv, flt cmaxv,
                               flt * bestcost, int * bestaxis, int * bestsplit) {
  bvh_bin bins[BVH_NUM_BINS];
  vector leftbmin[BVH_NUM_BINS], leftbmax[BVH_NUM_BINS];
  int leftcount[BVH_NUM_BINS];
  int i, k;

  if (cmaxv - cminv < EPSILON)
    return; /* degenerate on this axis, nothing to split */

  bin_reset(bins);
  for (i = start; i < end; i++) {
    int b = bin_index(vcomp(&prims[i].centroid, axis), cminv, cmaxv);
    bins[b].count++;
    box_union(&bins[b].bmin, &bins[b].bmax, &prims[i].bmin, &prims[i].bmax);
  }

  /* prefix scan left-to-right: cumulative bbox/count through bin k */
  {
    vector runmin, runmax;
    int runcount = 0;
    runmin.x = runmin.y = runmin.z =  FHUGE;
    runmax.x = runmax.y = runmax.z = -FHUGE;
    for (k = 0; k < BVH_NUM_BINS; k++) {
      if (bins[k].count > 0)
        box_union(&runmin, &runmax, &bins[k].bmin, &bins[k].bmax);
      runcount += bins[k].count;
      leftbmin[k] = runmin;
      leftbmax[k] = runmax;
      leftcount[k] = runcount;
    }
  }

  /* sweep right-to-left accumulating the right side, combine with the
   * left prefix computed above to get the SAH cost of splitting after
   * bin k (k = 0 .. BVH_NUM_BINS-2). */
  {
    vector runmin, runmax;
    int runcount = 0;
    runmin.x = runmin.y = runmin.z =  FHUGE;
    runmax.x = runmax.y = runmax.z = -FHUGE;
    for (k = BVH_NUM_BINS - 1; k > 0; k--) {
      if (bins[k].count > 0)
        box_union(&runmin, &runmax, &bins[k].bmin, &bins[k].bmax);
      runcount += bins[k].count;

      {
        int lc = leftcount[k - 1];
        int rc = runcount;
        if (lc > 0 && rc > 0) {
          flt cost = lc * box_surface_area(&leftbmin[k - 1], &leftbmax[k - 1]) +
                     rc * box_surface_area(&runmin, &runmax);
          if (cost < *bestcost) {
            *bestcost = cost;
            *bestaxis = axis;
            *bestsplit = k - 1;
          }
        }
      }
    }
  }
}

/* Partition prims[start,end) by axis/cminv/cmaxv/splitbin (the same
 * bin_index() rule evaluate_axis_sah used), in place. Returns the index of
 * the first "right side" element -- guaranteed to be in (start,end) since
 * evaluate_axis_sah only accepted candidates with lc>0 and rc>0. */
static int partition_by_bin(bvh_buildprim * prims, int start, int end,
                             int axis, flt cminv, flt cmaxv, int splitbin) {
  int i = start, j = end - 1;
  while (i <= j) {
    while (i <= j && bin_index(vcomp(&prims[i].centroid, axis), cminv, cmaxv) <= splitbin)
      i++;
    while (i <= j && bin_index(vcomp(&prims[j].centroid, axis), cminv, cmaxv) > splitbin)
      j--;
    if (i < j) {
      bvh_buildprim tmp = prims[i];
      prims[i] = prims[j];
      prims[j] = tmp;
      i++; j--;
    }
  }
  return i;
}

/* Fallback used only if SAH binning can't find a two-sided split (e.g. many
 * primitives sharing the same binned centroid): split by count at the
 * midpoint along the widest centroid axis. This always makes progress
 * (splits a range of >1 elements into two non-empty halves), guaranteeing
 * the recursion terminates regardless of how degenerate the input is. */
static int partition_median(bvh_buildprim * prims, int start, int end, int axis) {
  int mid = (start + end) / 2;
  int lo = start, hi = end - 1;
  /* simple selection (nth_element-lite): fine, this path is rare and n is
   * small by the time we fall back to it (it only runs on subranges where
   * SAH already found no useful split). */
  while (lo < hi) {
    flt pivot = vcomp(&prims[(lo + hi) / 2].centroid, axis);
    int i = lo, j = hi;
    while (i <= j) {
      while (vcomp(&prims[i].centroid, axis) < pivot) i++;
      while (vcomp(&prims[j].centroid, axis) > pivot) j--;
      if (i <= j) {
        bvh_buildprim tmp = prims[i]; prims[i] = prims[j]; prims[j] = tmp;
        i++; j--;
      }
    }
    if (mid <= j) hi = j;
    else if (mid >= i) lo = i;
    else break;
  }
  return mid;
}

static void make_leaf(bvhnode * node, int start, int n) {
  node->offset = start;
  node->nprims = (unsigned short) n;
}

static int bvh_build_node(bvh_buildstate * st, int start, int end, int depth) {
  int nodeidx = st->numnodes++;
  vector bmin, bmax, cmin, cmax;
  int n = end - start;

  compute_bounds(st->prims, start, end, &bmin, &bmax, &cmin, &cmax);
  st->nodes[nodeidx].bmin = bmin;
  st->nodes[nodeidx].bmax = bmax;

  if (n <= BVH_MAX_LEAF) {
    make_leaf(&st->nodes[nodeidx], start, n);
    return nodeidx;
  }

  /* SAH split quality has no balance guarantee -- a legitimately optimal
   * split can be arbitrarily lopsided (e.g. 1 vs n-1, for one outlier far
   * from a tight cluster), so worst-case depth from repeated SAH splits is
   * O(n), not O(log n). bvh_intersect's traversal stack pushes at most one
   * entry per interior node on the current path, so unbounded depth is an
   * unbounded stack write. Force a leaf once depth would put that at risk,
   * regardless of how many primitives are left -- a large, linearly-tested
   * leaf renders correctly (just slower) if this scene ever appears; every
   * scene tested during development stayed far under this cap. */
  if (depth >= BVH_MAX_DEPTH) {
    make_leaf(&st->nodes[nodeidx], start, n);
    return nodeidx;
  }

  {
    flt bestcost = FHUGE;
    int bestaxis = -1, bestsplit = -1;
    int splitidx, axis;

    evaluate_axis_sah(st->prims, start, end, 0, cmin.x, cmax.x, &bestcost, &bestaxis, &bestsplit);
    evaluate_axis_sah(st->prims, start, end, 1, cmin.y, cmax.y, &bestcost, &bestaxis, &bestsplit);
    evaluate_axis_sah(st->prims, start, end, 2, cmin.z, cmax.z, &bestcost, &bestaxis, &bestsplit);

    if (bestaxis < 0) {
      /* every axis was centroid-degenerate (all primitives share ~the
       * same position) -- SAH has nothing to work with. Leaf, however
       * large; this matches what a grid cell would do with coincident
       * objects (test them all), just without artificially forcing a
       * split that can't reduce cost. */
      make_leaf(&st->nodes[nodeidx], start, n);
      return nodeidx;
    }

    axis = bestaxis;
    {
      flt axcmin = vcomp(&cmin, axis), axcmax = vcomp(&cmax, axis);
      splitidx = partition_by_bin(st->prims, start, end, axis, axcmin, axcmax, bestsplit);
      if (splitidx == start || splitidx == end)
        splitidx = partition_median(st->prims, start, end, axis);
    }

    st->nodes[nodeidx].nprims = 0; /* mark interior (set after possible leaf
                                      * returns above, before recursing --
                                      * recursion reuses st->nodes, but this
                                      * node's own fields are already final
                                      * except offset, set below) */
    st->nodes[nodeidx].axis = (unsigned char) axis;

    bvh_build_node(st, start, splitidx, depth + 1);          /* left = nodeidx+1 */
    st->nodes[nodeidx].offset = bvh_build_node(st, splitidx, end, depth + 1); /* right */
  }

  return nodeidx;
}

int buildbvh_scene(scenedef * scene) {
  int numobj, i;
  object * cur, * next;
  object * remaining = NULL, ** remainingtail = &remaining;
  object * alllist = NULL;
  bvh_buildprim * buildprims;
  object ** primitives;
  bvh * b;
  bvh_buildstate st;
  char msgtxt[128];

  if (scene->objgroup.boundedobj == NULL)
    return 0;

  /* Same defensive split grid.c's engrid_object() does: most things on
   * boundedobj have a real bbox, but pull out anything that doesn't
   * (returns 0, e.g. an infinite/unbounded primitive that ended up on this
   * list) rather than assume it can't happen. */
  numobj = 0;
  for (cur = scene->objgroup.boundedobj; cur != NULL; cur = cur->nextobj)
    numobj++;

  buildprims = (bvh_buildprim *) malloc(numobj * sizeof(bvh_buildprim));

  i = 0;
  cur = scene->objgroup.boundedobj;
  while (cur != NULL) {
    next = cur->nextobj;
    {
      vector omin, omax;
      if (cur->methods->bbox((void *) cur, &omin, &omax)) {
        buildprims[i].bmin = omin;
        buildprims[i].bmax = omax;
        buildprims[i].centroid.x = 0.5 * (omin.x + omax.x);
        buildprims[i].centroid.y = 0.5 * (omin.y + omax.y);
        buildprims[i].centroid.z = 0.5 * (omin.z + omax.z);
        buildprims[i].obj = cur;
        i++;
        cur->nextobj = alllist;   /* now owned by the BVH's object list */
        alllist = cur;
      } else {
        *remainingtail = cur;
        remainingtail = (object **) &cur->nextobj;
      }
    }
    cur = next;
  }
  *remainingtail = NULL;
  numobj = i; /* actual bboxed count, may be < original if any were dropped */

  if (numobj == 0) {
    free(buildprims);
    /* nothing bounded to accelerate; put the list back unchanged */
    scene->objgroup.boundedobj = remaining;
    return 0;
  }

  st.prims = buildprims;
  st.numnodes = 0;
  st.maxnodes = 2 * numobj;
  st.nodes = (bvhnode *) malloc(st.maxnodes * sizeof(bvhnode));

  bvh_build_node(&st, 0, numobj, 0);

  primitives = (object **) malloc(numobj * sizeof(object *));
  for (i = 0; i < numobj; i++)
    primitives[i] = buildprims[i].obj;
  free(buildprims);

  b = (bvh *) malloc(sizeof(bvh));
  memset(b, 0, sizeof(bvh));
  b->methods = &bvh_methods;
  b->id = new_objectid(scene);
  b->primitives = primitives;
  b->objects = alllist;
  b->nodes = st.nodes;
  b->numnodes = st.numnodes;

  if (scene->verbosemode && scene->mynode == 0) {
    sprintf(msgtxt, "BVH: %d objects, %d nodes (SAH, %d-bin, leaf<=%d)",
            numobj, st.numnodes, BVH_NUM_BINS, BVH_MAX_LEAF);
    rt_ui_message(MSG_0, msgtxt);
  }

  b->nextobj = remaining;
  scene->objgroup.boundedobj = (object *) b;

  return 1;
}


/*
 * ---- Traversal ----
 */

static void bvh_intersect_leaf(const bvh * b, const bvhnode * node, ray * ry,
                                unsigned long serial) {
  object ** prims = &b->primitives[node->offset];
  int i;
#if !defined(DISABLEMBOX)
  unsigned long * mbox = ry->mbox;
#endif

  for (i = 0; i < node->nprims; i++) {
    object * obj = prims[i];
#if !defined(DISABLEMBOX)
    if (mbox[obj->id] != serial) {
      mbox[obj->id] = serial;
      obj->methods->intersect(obj, ry);
    }
#else
    obj->methods->intersect(obj, ry);
#endif
  }
}

/* Robust slab test (handles axis-aligned rays via IEEE INF semantics, same
 * trick as pbrt/most production BVHs); invd is 1/ry->d, precomputed once
 * per ray by the caller rather than once per node. Outputs tmin so the
 * caller can cache it (see the stack-with-tmin comment in bvh_intersect)
 * instead of redoing this whole test later just to react to ry->maxdist
 * having shrunk. */
static int bvh_node_hit(const bvhnode * node, const vector * o,
                         const vector * invd, flt maxdist, flt * outtmin) {
  flt tx1 = (node->bmin.x - o->x) * invd->x;
  flt tx2 = (node->bmax.x - o->x) * invd->x;
  flt tmin = MYMIN(tx1, tx2);
  flt tmax = MYMAX(tx1, tx2);

  flt ty1 = (node->bmin.y - o->y) * invd->y;
  flt ty2 = (node->bmax.y - o->y) * invd->y;
  tmin = MYMAX(tmin, MYMIN(ty1, ty2));
  tmax = MYMIN(tmax, MYMAX(ty1, ty2));

  flt tz1 = (node->bmin.z - o->z) * invd->z;
  flt tz2 = (node->bmax.z - o->z) * invd->z;
  tmin = MYMAX(tmin, MYMIN(tz1, tz2));
  tmax = MYMIN(tmax, MYMAX(tz1, tz2));

  *outtmin = tmin;
  return (tmax >= MYMAX(tmin, 0.0)) && (tmin <= maxdist);
}

#if !defined(USESINGLEFLT)
/*
 * SIMD (SSE2, packed double) test of both an interior node's children at
 * once -- the exact near+far pair bvh_intersect does at every interior
 * node. Profiling a 60K-object scene found this box test was ~50%+ of
 * total render time at ~47 calls/ray (measured via gprof's exact -pg call
 * counts, not sampling); the traversal was correct, just doing a lot of
 * cheap-looking-but-frequent scalar work. Rather than restructure the
 * tree into a 4-wide BVH (the usual way to get SIMD here), this fuses the
 * two DOUBLE-lane slab tests this build's `flt` (a double; see
 * tachyon.h's USESINGLEFLT switch) already needs into one SSE2 pass --
 * SSE2 is 2-wide for doubles and is baseline on every x86-64 CPU (part of
 * the ABI, unlike AVX/AVX2, which would give 4-wide but isn't universal
 * and is deliberately not what this build targets; see build_win/build.sh
 * for why -march=native etc. are avoided). Bit-identical to calling
 * bvh_node_hit twice: same multiplies/min/max in the same order per lane,
 * IEEE 754 guarantees SSE2 packed ops match their scalar equivalents.
 */
#include <emmintrin.h>

static void bvh_node_hit_pair(const bvhnode * na, const bvhnode * nb,
                               const vector * o, const vector * invd,
                               flt maxdist,
                               int * hita, flt * tmina,
                               int * hitb, flt * tminb) {
  __m128d ox = _mm_set1_pd(o->x), oy = _mm_set1_pd(o->y), oz = _mm_set1_pd(o->z);
  __m128d dx = _mm_set1_pd(invd->x), dy = _mm_set1_pd(invd->y), dz = _mm_set1_pd(invd->z);

  /* _mm_set_pd(hi, lo): lane 0 (low) = a, lane 1 (high) = b */
  __m128d bminx = _mm_set_pd(nb->bmin.x, na->bmin.x);
  __m128d bmaxx = _mm_set_pd(nb->bmax.x, na->bmax.x);
  __m128d bminy = _mm_set_pd(nb->bmin.y, na->bmin.y);
  __m128d bmaxy = _mm_set_pd(nb->bmax.y, na->bmax.y);
  __m128d bminz = _mm_set_pd(nb->bmin.z, na->bmin.z);
  __m128d bmaxz = _mm_set_pd(nb->bmax.z, na->bmax.z);

  __m128d tx1 = _mm_mul_pd(_mm_sub_pd(bminx, ox), dx);
  __m128d tx2 = _mm_mul_pd(_mm_sub_pd(bmaxx, ox), dx);
  __m128d tmin = _mm_min_pd(tx1, tx2);
  __m128d tmax = _mm_max_pd(tx1, tx2);

  __m128d ty1 = _mm_mul_pd(_mm_sub_pd(bminy, oy), dy);
  __m128d ty2 = _mm_mul_pd(_mm_sub_pd(bmaxy, oy), dy);
  tmin = _mm_max_pd(tmin, _mm_min_pd(ty1, ty2));
  tmax = _mm_min_pd(tmax, _mm_max_pd(ty1, ty2));

  __m128d tz1 = _mm_mul_pd(_mm_sub_pd(bminz, oz), dz);
  __m128d tz2 = _mm_mul_pd(_mm_sub_pd(bmaxz, oz), dz);
  tmin = _mm_max_pd(tmin, _mm_min_pd(tz1, tz2));
  tmax = _mm_min_pd(tmax, _mm_max_pd(tz1, tz2));

  {
    __m128d zero = _mm_setzero_pd();
    __m128d maxd = _mm_set1_pd(maxdist);
    __m128d tminclamped = _mm_max_pd(tmin, zero);
    __m128d hitmask = _mm_and_pd(_mm_cmpge_pd(tmax, tminclamped),
                                  _mm_cmple_pd(tmin, maxd));
    int mask = _mm_movemask_pd(hitmask);
    double tminarr[2];
    _mm_storeu_pd(tminarr, tmin);

    *hita = mask & 1;
    *hitb = (mask >> 1) & 1;
    *tmina = tminarr[0];
    *tminb = tminarr[1];
  }
}
#endif /* !USESINGLEFLT */

typedef struct {
  int idx;
  flt tmin;
} bvh_stackentry;

static void bvh_intersect(const bvh * b, ray * ry) {
  bvh_stackentry stack[BVH_STACK_SIZE];
  int sp = 0;
  int nodeidx;
  unsigned long serial;
  vector invd;
  flt rd[3];
  flt tmin;

  if (ry->flags & RT_RAY_FINISHED)
    return;

  if (b->numnodes == 0)
    return;

  invd.x = 1.0 / ry->d.x;
  invd.y = 1.0 / ry->d.y;
  invd.z = 1.0 / ry->d.z;
  rd[0] = ry->d.x; rd[1] = ry->d.y; rd[2] = ry->d.z;

  serial = ry->serial;

  if (!bvh_node_hit(&b->nodes[0], &ry->o, &invd, ry->maxdist, &tmin))
    return;

  nodeidx = 0;
  for (;;) {
    const bvhnode * node = &b->nodes[nodeidx];

    if (node->nprims > 0) {
      bvh_intersect_leaf(b, node, ry, serial);
      if (ry->flags & RT_RAY_FINISHED)
        return;
    } else {
      int left = nodeidx + 1;
      int right = node->offset;
      int near, far;
      flt neartmin, fartmin;
      int hitnear, hitfar;

      if (rd[node->axis] < 0.0) {
        near = right; far = left;
      } else {
        near = left; far = right;
      }

      /* Stash far's tmin (the ray parameter where it enters the box) along
       * with its index. ry->maxdist can only shrink between now and when
       * we pop this entry (a closer hit found elsewhere), and the box
       * geometry doesn't change -- so "is this box still worth visiting"
       * later is exactly "tmin <= ry->maxdist", no need to redo the whole
       * 6-multiply slab test on pop. This was the actual hot spot: ~60
       * full box tests/ray, about half of them this exact redundant
       * retest, ~53% of total render time on a 60K-object scene. The
       * near+far test below is the other half -- see bvh_node_hit_pair(). */
#if !defined(USESINGLEFLT)
      bvh_node_hit_pair(&b->nodes[near], &b->nodes[far], &ry->o, &invd, ry->maxdist,
                         &hitnear, &neartmin, &hitfar, &fartmin);
#else
      hitfar = bvh_node_hit(&b->nodes[far], &ry->o, &invd, ry->maxdist, &fartmin);
      hitnear = bvh_node_hit(&b->nodes[near], &ry->o, &invd, ry->maxdist, &neartmin);
#endif

      if (hitfar) {
        stack[sp].idx = far;
        stack[sp].tmin = fartmin;
        sp++;
      }

      if (hitnear) {
        nodeidx = near;
        continue; /* descend without touching the stack */
      }
    }

    /* Nothing more to descend into from here (leaf we just tested, or
     * interior whose near child missed): pop the next still-live node. */
    for (;;) {
      if (sp == 0)
        return;
      sp--;
      if (stack[sp].tmin <= ry->maxdist) {
        nodeidx = stack[sp].idx;
        break;
      }
    }
  }
}
