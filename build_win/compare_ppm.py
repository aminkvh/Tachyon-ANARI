#!/usr/bin/env python3
"""compare_ppm.py A.ppm B.ppm - per-image mean/stddev plus mean/max absolute
pixel difference between two binary (P6) PPM files of identical size.
Used to statistically verify the jitter_sphere3f AO sampler rewrite, where
output is expected to differ pixel-by-pixel (different noise pattern) but
should have matching mean brightness (no bias) at high sample counts.
"""
import sys


def read_ppm(path):
    with open(path, "rb") as f:
        data = f.read()
    assert data[:2] == b"P6", f"{path}: not a binary PPM"
    pos = 2
    vals = []
    while len(vals) < 3:
        while data[pos:pos+1].isspace():
            pos += 1
        if data[pos:pos+1] == b"#":
            while data[pos:pos+1] != b"\n":
                pos += 1
            continue
        start = pos
        while not data[pos:pos+1].isspace():
            pos += 1
        vals.append(int(data[start:pos]))
    pos += 1  # single whitespace byte after maxval
    w, h, maxval = vals
    pixels = data[pos:pos + w*h*3]
    return w, h, pixels


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    wa, ha, pa = read_ppm(a_path)
    wb, hb, pb = read_ppm(b_path)
    assert (wa, ha) == (wb, hb), "size mismatch"

    n = len(pa)
    sum_a = sum(pa)
    sum_b = sum(pb)
    mean_a = sum_a / n
    mean_b = sum_b / n

    var_a = sum((x - mean_a) ** 2 for x in pa) / n
    var_b = sum((x - mean_b) ** 2 for x in pb) / n

    diffs = [abs(x - y) for x, y in zip(pa, pb)]
    mean_diff = sum(diffs) / n
    max_diff = max(diffs)
    identical = sum(1 for d in diffs if d == 0)

    print(f"  {a_path}: mean={mean_a:.3f} std={var_a**0.5:.3f}")
    print(f"  {b_path}: mean={mean_b:.3f} std={var_b**0.5:.3f}")
    print(f"  mean|diff|={mean_diff:.3f}  max|diff|={max_diff}  "
          f"identical_bytes={100.0*identical/n:.1f}%")


if __name__ == "__main__":
    main()
