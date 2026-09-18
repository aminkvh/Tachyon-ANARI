#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p build_wsl/prof/obj build_wsl/prof/bin build_wsl/prof/out
CFLAGS="-m64 -O2 -pg -g -Wall -DTHR -DUSEATOMICBARRIERS -D_REENTRANT -Isrc -Idemosrc -DLinux"
LIBSRCS="api apigeom box global hash parallel threads camera render trace grid intersect sphere plane ring triangle cylinder quadric extvol vol ui util vector shade texture coordsys imap light imageio jpeg pngfile ppm psd sgirgb tgafile winbmp"
DEMOSRCS="main getargs parse nffparse ac3dparse trackball"

for f in $LIBSRCS; do
  echo "compiling src/$f.c"
  gcc $CFLAGS -c "src/$f.c" -o "build_wsl/prof/obj/$f.o"
done

for f in $DEMOSRCS; do
  echo "compiling demosrc/$f.c"
  gcc $CFLAGS -c "demosrc/$f.c" -o "build_wsl/prof/obj/$f.o"
done

echo "compiling stub_spaceball"
gcc $CFLAGS -c build_win/stub_spaceball.c -o build_wsl/prof/obj/stub_spaceball.o

ALLOBJS=""
for f in $LIBSRCS $DEMOSRCS stub_spaceball; do
  ALLOBJS="$ALLOBJS build_wsl/prof/obj/$f.o"
done

echo "linking"
gcc -m64 -O2 -pg $ALLOBJS -o build_wsl/prof/bin/tachyon -lpthread -lm

echo "DONE"
ls -la build_wsl/prof/bin/
