/*
 * bvh.h - bounding volume hierarchy acceleration structure, built with
 *         binned surface-area-heuristic (SAH) splits.
 *
 * Alternative to grid.c's uniform grid. Same object_methods interface
 * (intersect/bbox/free), so it drops into scene->objgroup.boundedobj the
 * same way a grid object does; see buildbvh_scene() / render.c.
 */

int buildbvh_scene(scenedef * scene);

#ifdef BVH_PRIVATE

/* Binned SAH build-time tuning. Not exposed on the CLI: unlike the grid's
 * boundthresh, these don't correspond to a concept a scene author would
 * reasonably want to tune per-scene. */
#define BVH_NUM_BINS     12
#define BVH_MAX_LEAF     4    /**< stop splitting at/below this many prims */
#define BVH_STACK_SIZE   128  /**< traversal stack; tree depth is O(log N)
                                *  for a balanced SAH tree, but the degenerate
                                *  median-split fallback (see bvh.c) only
                                *  guarantees non-empty halves, not balance,
                                *  so worst-case depth is O(numprims) without
                                *  BVH_MAX_DEPTH below forcing a leaf first. */
#define BVH_MAX_DEPTH    96   /**< builder depth cap, comfortably under
                                *  BVH_STACK_SIZE -- enforced in the builder
                                *  so the traversal stack bound holds by
                                *  construction, not by hoping trees stay
                                *  balanced */

/* Linearized (array-of-structs, not pointer tree) node layout, in DFS
 * pre-order: an interior node's left child is always node[i+1]; its right
 * child is at node[i].secondChildOffset. This is the standard compact BVH
 * layout (as in pbrt's LinearBVHNode) -- no pointers to chase, and every
 * node is one cache line-friendly struct. */
typedef struct {
  vector bmin;
  vector bmax;
  int offset;              /**< interior: index of right child (2nd child).
                              *  leaf: index into bvh->primitives[]        */
  unsigned short nprims;   /**< 0 => interior node; >0 => leaf, this many
                              *  primitives starting at primitives[offset] */
  unsigned char axis;      /**< split axis (0/1/2); interior nodes only,
                              *  used to pick a near-child traversal order */
} bvhnode;

typedef struct {
  RT_OBJECT_HEAD
  bvhnode * nodes;
  int numnodes;
  object ** primitives;   /**< flat, BVH-ordered array of leaf object ptrs */
  object * objects;       /**< all objects owned by this BVH (nextobj-linked,
                             *  for free_objects()); same role as grid's   */
} bvh;

static int bvh_bbox(void * obj, vector * min, vector * max);
static void bvh_free(void * v);
static void bvh_intersect(const bvh *, ray *);

#endif
