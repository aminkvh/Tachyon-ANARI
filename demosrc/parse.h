/*
 * parse.h - this file contains defines for model file reading.
 *
 *  $Id: parse.h,v 1.32 2011/02/15 23:51:58 johns Exp $
 */

#define PARSENOERR       0
#define PARSEBADFILE     1
#define PARSEBADSUBFILE  2
#define PARSEBADSYNTAX   4
#define PARSEEOF         8
#define PARSEALLOCERR    16
 
unsigned int readmodel(const char *, SceneHandle);

#ifdef PARSE_INTERNAL
#define TEXNAMELEN 255

typedef struct {
        char * name;
        void * tex;
} texentry;

/* key for sharing identical constant-color textures */
typedef struct {
  float v[11];           /* amb dif spec opac phong size outl outlw rgb */
  int phongtype;
  int transmode;
} texkey;

typedef struct {
  texkey key;
  void * tex;            /* NULL for an empty slot                  */
} texcacheent;

/* spheres of one texture, emitted as a sphere array at end of scene */
typedef struct {
  void * tex;
  int n;
  int cap;
  float * ctr;           /* 3 floats per sphere */
  float * rad;
} sphbatch;

typedef struct {
  char * buf;            /* whole current input file, read in one shot   */
  long buflen;            /* length of buf, in bytes                      */
  long bufpos;            /* current read cursor into buf                 */
  const char * filename; /* filename of current input file */
  texentry *textable;    /* texture lookup table */
  texentry defaulttex;   /* The default texture when a lookup fails */
  int numtextures;       /* number of TEXDEF textures               */
  int maxtextures;       /* number of TEXDEF textures               */
  int numobjectsparsed;  /* total number of objects parsed so far   */
  rt_hash_t texhash;     /* hash table for texture name lookup      */
  int transmode;         /* transparency rendering mode flags       */
  texcacheent * texcache; /* hash of shareable textures             */
  int texcachesz;        /* slots in texcache (power of 2, or 0)    */
  int texcachecnt;       /* used slots                              */
  sphbatch * batches;    /* pending sphere batches                  */
  int numbatches;
  int maxbatches;
  int lastbatch;         /* batch used by the previous sphere       */
  int * bhash;           /* tex pointer -> batch index, -1 if empty */
  int bhashsz;           /* power of 2, or 0                        */
} parsehandle;

typedef struct {
   double rx1; double rx2; double rx3;
   double ry1; double ry2; double ry3;
   double rz1; double rz2; double rz3;
} RotMat;

typedef unsigned int errcode;

/* texture lookup table code */
static void reset_tex_table(parsehandle *, SceneHandle);
static void free_tex_table(parsehandle *, SceneHandle);
static errcode add_texture(parsehandle *, void * tex, const char *);
static void * find_texture(parsehandle *, const char *);

/* scene file parsing code */
static errcode GetString(parsehandle *, const char *);
static errcode GetScenedefs(parsehandle *, SceneHandle);
static errcode GetCamera(parsehandle *, SceneHandle);
static errcode GetColor(parsehandle *, apicolor *);
/* static errcode GetFloat(parsehandle *, apiflt *); */
static errcode GetVector(parsehandle *, apivector *);
static errcode GetTexDef(parsehandle *, SceneHandle);
static errcode GetImageDef(parsehandle *, SceneHandle);
static errcode GetTexAlias(parsehandle *);
static errcode GetTexture(parsehandle *, SceneHandle, void **);
void * GetTexBody(parsehandle *, SceneHandle, int);
static errcode GetBackGnd(parsehandle *, SceneHandle);
static errcode GetBackGndGradient(parsehandle *, SceneHandle);
static errcode GetFog(parsehandle *, SceneHandle);
static errcode GetCylinder(parsehandle *, SceneHandle);
static errcode GetFCylinder(parsehandle *, SceneHandle);
static errcode GetPolyCylinder(parsehandle *, SceneHandle);
static errcode GetSphere(parsehandle *, SceneHandle);
static void flush_spheres(parsehandle *, SceneHandle);
static errcode GetSphereArray(parsehandle *, SceneHandle);
static errcode GetPlane(parsehandle *, SceneHandle);
static errcode GetRing(parsehandle *, SceneHandle);
static errcode GetBox(parsehandle *, SceneHandle);
static errcode GetVol(parsehandle *, SceneHandle);
static errcode GetTri(parsehandle *, SceneHandle);
static errcode GetSTri(parsehandle *, SceneHandle);
static errcode GetVCSTri(parsehandle *, SceneHandle);
static errcode GetVertexArray(parsehandle *, SceneHandle);
static errcode GetDirLight(parsehandle *, SceneHandle);
static errcode GetLight(parsehandle *, SceneHandle);
static errcode GetSkyLight(parsehandle *, SceneHandle);
static errcode GetSpotLight(parsehandle *, SceneHandle);
static errcode GetLandScape(parsehandle *, SceneHandle);
static errcode GetTPolyFile(parsehandle *, SceneHandle);
static errcode GetObject(parsehandle *, SceneHandle);
static errcode ReadIncludeFile(parsehandle *, const char *, SceneHandle);
static errcode GetClipGroup(parsehandle * ph, SceneHandle scene);
static errcode GetClipGroupEnd(parsehandle * ph, SceneHandle scene);

#ifdef USELIBMGF
static errcode GetMGFFile(parsehandle *, SceneHandle);
#endif


/* list of all keywords the parser looks for */
#if 0
ATTENUATION
BACKGROUND
BOX
CAMERA
COLOR
CYLINDER
DATA
DIRECTIONAL_LIGHT
END_CAMERA
END_CLIPGROUP
END_SCENE
EXP
EXP2OFF
FCYLINDER
FISHEYE
FOG
FRUSTUM
INCLUDE
KIDS
LIGHT
LINEAR
LOC
MAT
MATERIAL
METAL
MGFFILE
NAME
NUMPLANES
NUMSURF
NUMVERT
OBJECT
ORTHOGRAPHIC
PERSPECTIVE
PERSPECTIVE_DOF
PHONG
PLANE
PLASTIC
POLYCYLINDER
PROJECTION
REFS
RING
ROT
SCALARVOL
SCAPE
SPHERE
SPOTLIGHT
START_CLIPGROUP
STRI
SURF
TEXALIAS
TEXDEF
TEXREP
TEXTURE
TPOLYFILE
TRI
URL
VCST
VCSTRI
VERTEXARRAY
  COORDS
  COLORS
  NORMALS
  FACETS
  NUMVERTS
  NUMFACETS
ZOOM
#endif

#endif

