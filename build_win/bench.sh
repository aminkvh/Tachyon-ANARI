#!/bin/sh
# bench.sh - baseline/regression benchmark + correctness harness.
#
# Renders a fixed set of scenes spanning Tachyon's scene classes at a fixed
# resolution and thread count, 3 times each, and saves the images. Used two
# ways:
#   sh build_win/bench.sh baseline   -> saves to build_win/bench/baseline/
#   sh build_win/bench.sh current    -> saves to build_win/bench/current/,
#                                        then diffs pixel-for-pixel against
#                                        baseline/ with ImageMagick if present
#
# This is the only correctness gate in this repo (no test suite): any
# optimization must produce byte-identical (or explicitly-tolerance-bounded)
# images versus the pre-optimization baseline.

set -e
cd "$(dirname "$0")/.."

TAG="${1:-current}"
OUTDIR="build_win/bench/$TAG"
mkdir -p "$OUTDIR"

BIN=build_win/bin/tachyon.exe
RES="512 512"
THREADS=8
RUNS=3

SCENES="teapot skull 820spheres sc98ao trypsin4pti"

echo "scene,run,seconds" > "$OUTDIR/timings.csv"

for s in $SCENES; do
  for i in $(seq 1 $RUNS); do
    START=$(date +%s.%N)
    "$BIN" "scenes/$s.dat" -res $RES -numthreads $THREADS -format PPM -o "$OUTDIR/$s.ppm" > "$OUTDIR/$s.run$i.log" 2>&1
    END=$(date +%s.%N)
    SEC=$(awk "BEGIN{printf \"%.4f\", $END-$START}")
    echo "$s,$i,$SEC" >> "$OUTDIR/timings.csv"
    echo "$s run $i: ${SEC}s"
  done
done

echo ""
echo "=== Ray-tracing-time (from stdout) summary ==="
for s in $SCENES; do
  grep "Ray Tracing Time" "$OUTDIR/$s.run"*.log
done

if [ "$TAG" != "baseline" ] && [ -d build_win/bench/baseline ]; then
  echo ""
  echo "=== Pixel diff vs baseline ==="
  for s in $SCENES; do
    if command -v compare >/dev/null 2>&1; then
      compare -metric AE "build_win/bench/baseline/$s.ppm" "$OUTDIR/$s.ppm" null: 2>&1 | sed "s/^/$s: /"
      echo ""
    else
      cmp -s "build_win/bench/baseline/$s.ppm" "$OUTDIR/$s.ppm" && echo "$s: byte-identical" || echo "$s: DIFFERS (install ImageMagick 'compare' for a pixel diff count)"
    fi
  done
fi
