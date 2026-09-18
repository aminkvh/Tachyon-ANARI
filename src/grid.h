/*
 * grid.h - spatial subdivision efficiency structures
 *
 * $Id: grid.h,v 1.17 2011/02/05 08:10:11 johns Exp $
 * 
 */

int engrid_scene(scenedef * scene, int boundthresh);
object * newgrid(scenedef * scene, int xsize, int ysize, int zsize, 
                 vector min, vector max);

#ifdef GRID_PRIVATE

typedef struct objectlist {
  struct objectlist * next; /**< next link in the list */
  object * obj;             /**< the actual object     */
} objectlist; 

typedef struct {
  RT_OBJECT_HEAD
  int xsize;           /**< number of cells along the X direction */
  int ysize;           /**< number of cells along the Y direction */
  int zsize;           /**< number of cells along the Z direction */
  vector min;          /**< minimum coords for the box containing the grid */
  vector max;          /**< maximum coords for the box containing the grid */
  vector voxsize;      /**< the size of a grid cell/voxel */
  object * objects;    /**< all objects contained in the grid */

  /** Build-time per-cell linked lists. Populated by engrid_object() while
   *  the grid (and any subgrids carved out of overfull cells) is being
   *  assembled, then converted to the flat cellobjs/cellstart arrays below
   *  by grid_freeze() and released. NULL once frozen. */
  objectlist ** buildcells;

  /** Post-freeze flattened cell contents (CSR layout): the objects in
   *  voxel i are cellobjs[cellstart[i] .. cellstart[i+1]). This is what
   *  grid_intersect() walks at render time -- a contiguous array instead
   *  of a linked list, so cell traversal is sequential/prefetch-friendly
   *  instead of chasing individually-malloc'd nodes. */
  object ** cellobjs;
  int * cellstart;     /**< xsize*ysize*zsize+1 entries */
} grid;

typedef struct {
  int x;               /**< Voxel X address */
  int y;               /**< Voxel Y address */
  int z;               /**< Voxel Z address */
} gridindex; 

/** Convert from voxel index along X/Y/Z to corresponding coordinate.  */
#define voxel2x(g,X)  ((X) * (g->voxsize.x) + (g->min.x))
/** Convert from voxel index along X/Y/Z to corresponding coordinate.  */
#define voxel2y(g,Y)  ((Y) * (g->voxsize.y) + (g->min.y))
/** Convert from voxel index along X/Y/Z to corresponding coordinate.  */
#define voxel2z(g,Z)  ((Z) * (g->voxsize.z) + (g->min.z))

/** Convert from voxel coordinate to corresponding indices.  */
#define x2voxel(g,x)            (((x) - g->min.x) / g->voxsize.x)
/** Convert from voxel coordinate to corresponding indices.  */
#define y2voxel(g,y)            (((y) - g->min.y) / g->voxsize.y)
/** Convert from voxel coordinate to corresponding indices.  */
#define z2voxel(g,z)            (((z) - g->min.z) / g->voxsize.z)


static void gridstats(int xs, int ys, int zs, int numobj);
static int grid_bbox(void * obj, vector * min, vector * max);
static void grid_free(void * v);
static void grid_freeze(grid * g);

static int cellbound(const grid *g, const gridindex *index, vector * cmin, vector * cmax);

static int engrid_objlist(grid * g, object ** list);
static int engrid_object(grid * g, object * obj, int addtolist);

static int engrid_objectlist(grid * g, objectlist ** list);
static int engrid_cell(scenedef *, int, grid *, gridindex *);

static int pos2grid(grid * g, vector * pos, gridindex * index);
static void grid_test_cell(const grid * g, int voxindex, ray * ry, unsigned long serial);
static void grid_intersect(const grid *, ray *);
static int grid_bounds_intersect(const grid * g, const ray * ry, flt *hitnear, flt *hitfar);

#endif


