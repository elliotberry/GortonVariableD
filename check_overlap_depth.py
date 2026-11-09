#!/usr/bin/env python3
"""
Report how many contours overlap at the most crowded point of each glyph.

The variable font is built from deliberately overlapping contours.  CoreText's
rasterizer (Safari, Chrome and every other app on macOS/iOS) keeps its winding
count in four bits, so where 16 or more contours stack up the count wraps to
zero and the glyph shows holes and stray hairlines.  Keep every glyph well under
that: this script fails if any glyph has more than --limit layers (default 14) at any
of the master pen radii.

    python3 check_overlap_depth.py [--limit 12]

Needs numpy and Pillow (pip install numpy pillow).
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

import gorton_stroke as gs

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = 4.0  # font units per pixel
MASTER_RADII = (105.0, 140.0, 157.5, 210.0)  # Light, Regular, Medium, Heavy pens


def flatten(contour, steps=12):
    pts, cur = [], None
    for seg in contour:
        if seg[0] in ("M", "L"):
            cur = seg[1]
            pts.append(cur)
        else:
            for i in range(1, steps + 1):
                pts.append(gs.bez((cur, seg[1], seg[2], seg[3]), i / steps))
            cur = seg[3]
    return pts


def max_depth(contours):
    polys = [flatten(c) for c in contours]
    if not polys:
        return 0
    xs = [p[0] for poly in polys for p in poly]
    ys = [p[1] for poly in polys for p in poly]
    x0, y0 = min(xs) - 4, min(ys) - 4
    w = int((max(xs) - x0) / SCALE) + 3
    h = int((max(ys) - y0) / SCALE) + 3
    depth = np.zeros((h, w), dtype=np.int16)
    for poly in polys:
        img = Image.new("1", (w, h), 0)
        ImageDraw.Draw(img).polygon([((x - x0) / SCALE, (y - y0) / SCALE) for x, y in poly], fill=1)
        depth += np.asarray(img, dtype=np.int16)
    return int(depth.max())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=14)
    ap.add_argument("--skeleton", default=os.path.join(HERE, "skeleton.json"))
    args = ap.parse_args(argv)

    with open(args.skeleton) as f:
        skeleton = json.load(f)

    worst, failures = (0, None, None), []
    for name, glyph in skeleton["glyphs"].items():
        if not glyph["contours"]:
            continue
        ops = gs.plan_glyph(glyph["contours"], 105, 210)
        for r in MASTER_RADII:
            d = max_depth(gs.render_plan(ops, r, 0.0))
            if d > worst[0]:
                worst = (d, name, r)
            if d > args.limit:
                failures.append((name, r, d))
    print("deepest overlap: %d layers (glyph %s, pen radius %g)" % worst)
    for name, r, d in failures:
        print("  TOO DEEP: %s at r=%g: %d layers" % (name, r, d))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
