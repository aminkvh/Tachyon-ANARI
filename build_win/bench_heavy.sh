#!/bin/sh
# bench_heavy.sh - headline performance numbers at a realistic resolution.
#
# build_win/bench.sh (512x512, no AA) is too small to produce a stable wall
# clock on the lighter scenes; this renders at 1600x1200 with 4x
# supersampling on the two scenes that represent real Tachyon workloads:
# sc98ao (ambient-occlusion/shadow-ray bound) and trypsin4pti (molecular,
# Tachyon's actual target use case in VMD). Also emits a single-threaded
# run of each, for use with a profiler (thread scheduling noise makes
# multithreaded profiles hard to read).
#
# Usage: sh build_win/bench_heavy.sh <tag>

set -e
cd "$(dirname "$0")/.."

TAG="${1:-current}"
OUTDIR="build_win/bench/heavy-$TAG"
mkdir -p "$OUTDIR"

BIN=build_win/bin/tachyon.exe
RES="1600 1200"
SCENES="sc98ao trypsin4pti"

echo "scene,threads,seconds" > "$OUTDIR/timings.csv"

for s in $SCENES; do
  for T in 1 16; do
    START=$(date +%s.%N)
    "$BIN" "scenes/$s.dat" -res $RES -aasamples 4 -numthreads $T -format PPM -o "$OUTDIR/$s.t$T.ppm" > "$OUTDIR/$s.t$T.log" 2>&1
    END=$(date +%s.%N)
    SEC=$(awk "BEGIN{printf \"%.4f\", $END-$START}")
    echo "$s,$T,$SEC" >> "$OUTDIR/timings.csv"
    echo "$s (threads=$T): ${SEC}s wall"
    grep "Ray Tracing Time" "$OUTDIR/$s.t$T.log" | sed 's/^/  /'
  done
done
