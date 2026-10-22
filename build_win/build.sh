#!/bin/sh
# build.sh - headless Windows/MinGW-w64 build of the Tachyon CLI renderer.
#
# There is no upstream win64 target using a modern MinGW toolchain (the
# unix/Make-arch win32 target is a 1990s Cygwin/egcs recipe), so this is a
# standalone build script rather than a patch to the autoconf-less unix/
# Make-* chain. It builds the core ray tracing library plus the demosrc
# command-line front end (main.c/parse.c/getargs.c/nffparse.c/ac3dparse.c),
# skipping the pieces that need libraries this box doesn't have:
#   - libjpeg/libpng (jpeg.c/pngfile.c compile to IMAGEUNSUP stubs without
#     -DUSEJPEG/-DUSEPNG, so TGA/PPM/PSD/BMP/RGB output still work)
#   - MGF/OpenGL/X11 (main.c gates these behind USELIBMGF/USEOPENGL)
#   - spaceball.c/glwin.c interactive OpenGL fly-through mode, replaced by
#     stub_spaceball.c since it's unreachable from the headless CLI path
#
# Usage: run from the Tachyon-dev repo root (git bash / MSYS):
#   sh build_win/build.sh [extra CFLAGS...]
# e.g. sh build_win/build.sh -DNDEBUG

set -e
cd "$(dirname "$0")/.."

OBJDIR=build_win/obj
BINDIR=build_win/bin
mkdir -p "$OBJDIR" "$BINDIR"

# -flto lets the optimizer inline/CSE across translation unit boundaries
# (e.g. the primitive intersectors calling into shade.c/intersect.c).
# Deliberately no -march=native / -mavx2 / -ffast-math here: this build is
# meant to run on whatever x86-64 CPU it's copied to, and -ffast-math
# changes pixel values (unsafe FP reassociation), which fails the
# byte-identical bar the rest of this optimization work is held to.
# Also builds on Linux and macOS (x86-64 or ARM64): only the linker libs
# and executable name differ per platform.
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) EXE=tachyon.exe; LIBS="-lpthread -lm -lws2_32" ;;
  *)                    EXE=tachyon;     LIBS="-lpthread -lm" ;;
esac
CC="${CC:-gcc}"

BASE_CFLAGS="-O3 -flto -Wall -DUSEPOSIXTHREADS -DTHR -DUSEATOMICBARRIERS -D_REENTRANT -Isrc -Idemosrc"
CFLAGS="$BASE_CFLAGS $*"

LIBSRCS="api apigeom box global hash parallel threads camera render trace grid bvh intersect sphere spherearray plane ring triangle cylinder quadric extvol vol ui util vector shade texture coordsys imap light imageio jpeg pngfile ppm psd sgirgb tgafile winbmp"
DEMOSRCS="main getargs parse nffparse ac3dparse trackball"

for f in $LIBSRCS; do
  $CC $CFLAGS -c "src/$f.c" -o "$OBJDIR/$f.o"
done

for f in $DEMOSRCS; do
  $CC $CFLAGS -c "demosrc/$f.c" -o "$OBJDIR/$f.o"
done

$CC $CFLAGS -c build_win/stub_spaceball.c -o "$OBJDIR/stub_spaceball.o"

ALLOBJS=""
for f in $LIBSRCS $DEMOSRCS stub_spaceball; do
  ALLOBJS="$ALLOBJS $OBJDIR/$f.o"
done

$CC -O3 -flto $ALLOBJS -o "$BINDIR/$EXE" $LIBS

echo "Built $BINDIR/$EXE"
