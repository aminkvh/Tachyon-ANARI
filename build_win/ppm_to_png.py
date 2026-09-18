#!/usr/bin/env python3
"""ppm_to_png.py A.ppm [B.ppm ...] - convert binary PPM (P6) renders to PNG
for display (browsers/artifacts can't render PPM directly)."""
import sys
from PIL import Image

for path in sys.argv[1:]:
    img = Image.open(path)
    out = path.rsplit(".", 1)[0] + ".png"
    img.save(out)
    print(f"{path} -> {out} ({img.size[0]}x{img.size[1]})")
