#!/bin/sh
# bench_ao_convergence.sh - statistical verification for the jitter_sphere3f
# AO sampler rewrite (cube-rejection -> disk-rejection, both uniform on the
# sphere but a different RNG-call sequence, so output is no longer
# byte-identical). Renders the AO scene at increasing skylight sample
# counts with two binaries and reports per-image mean/stddev plus
# mean/max absolute pixel difference between them, via compare_ppm.py.
#
# Usage: sh build_win/bench_ao_convergence.sh <old.exe> <new.exe>

set -e
cd "$(dirname "$0")/.."

OLD="${1:-build_win/bin_old/tachyon.exe}"
NEW="${2:-build_win/bin_new/tachyon.exe}"
OUTDIR=build_win/bench/ao_convergence
mkdir -p "$OUTDIR"

for N in 16 128 1024; do
  "$OLD" scenes/sc98ao.dat -res 400 300 -numthreads 1 -skylight_samples $N -format PPM -o "$OUTDIR/old_$N.ppm" > "$OUTDIR/old_$N.log" 2>&1
  "$NEW" scenes/sc98ao.dat -res 400 300 -numthreads 1 -skylight_samples $N -format PPM -o "$OUTDIR/new_$N.ppm" > "$OUTDIR/new_$N.log" 2>&1
  echo "=== N=$N samples ==="
  python3 build_win/compare_ppm.py "$OUTDIR/old_$N.ppm" "$OUTDIR/new_$N.ppm"
done
