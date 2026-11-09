#!/usr/bin/env fontforge -lang=py -script
"""
Export the stroke "skeleton" (the centre-line template) from gortondigital.sfd
to a plain JSON file, so the rest of the variable-font build does not need
FontForge.

Usage:
    fontforge -lang=py -script export_skeleton.py gortondigital.sfd skeleton.json

For every glyph that would be output, the JSON holds the advance width of the
template, its unicode value, and its contours as lists of
[x, y, is_on_curve] points (cubic splines, exactly as FontForge stores them).
References are unlinked so every glyph carries its own contours.
"""
import json
import sys

import fontforge


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "gortondigital.sfd"
    dst = sys.argv[2] if len(sys.argv) > 2 else "skeleton.json"

    font = fontforge.open(src)
    if font.is_quadratic:
        raise SystemExit("skeleton must be cubic (PostScript) splines")

    glyphs = {}
    order = []
    for glyph in font.glyphs():
        if not glyph.isWorthOutputting():
            continue
        glyph.unlinkRef()
        contours = []
        for contour in glyph.foreground:
            contours.append(
                {
                    "closed": bool(contour.closed),
                    "points": [[p.x, p.y, bool(p.on_curve)] for p in contour],
                }
            )
        glyphs[glyph.glyphname] = {
            "unicode": glyph.unicode if glyph.unicode > 0 else None,
            "width": glyph.width,
            "contours": contours,
        }
        order.append(glyph.glyphname)

    data = {
        "unitsPerEm": font.em,
        "ascent": font.ascent,
        "descent": font.descent,
        "order": order,
        "glyphs": glyphs,
    }
    with open(dst, "w") as fh:
        json.dump(data, fh, indent=1)
        fh.write("\n")
    print("Wrote %s (%d glyphs)" % (dst, len(order)))


main()
