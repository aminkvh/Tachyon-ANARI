/*
 * stub_spaceball.c - headless Windows CLI build stub.
 *
 * The real spaceball.c/glwin.c pair only matters for the interactive
 * OpenGL "fly-through" mode (-spaceball flag), which needs an OpenGL
 * window we don't build here. This headless batch renderer never sets
 * opt.spaceballon, so fly_scene() is unreachable, but main.o still
 * references these two symbols at link time. Stub them out.
 */
#include <stddef.h>
#include "tachyon.h"
#include "spaceball.h"

void * tachyon_init_spaceball(SceneHandle scene, void * glwin, char * port) {
  (void)scene; (void)glwin; (void)port;
  return NULL;
}

int tachyon_spaceball_update(sbHandle * bh, SceneHandle scene) {
  (void)bh; (void)scene;
  return 1; /* "done" */
}
