#!/usr/bin/env python3
"""
Build the Gorton Digital variable font (wght axis 300-700).

    python3 build_variable.py            # skeleton -> masters -> VF (TTF + CFF2 OTF)
    python3 build_variable.py --masters  # only regenerate the four UFO masters

Pipeline
--------
1. export_skeleton.py   (FontForge)  gortondigital.sfd  ->  skeleton.json
2. gorton_stroke.py     (pure Python) skeleton.json     ->  gortondigital*.ufo
3. fontmake             GortonDigital.designspace       ->  GortonDigital-VF.ttf/.otf

Step 1 is only re-run when gortondigital.sfd is newer than skeleton.json (and
FontForge is available), so everything after it works without FontForge.

Why not stroke with FontForge, as generateStrokes.ff does for the static fonts?
FontForge's stroker removes overlaps internally, so each weight ends up with a
different outline structure, and the masters cannot be interpolated.  The
stroker in gorton_stroke.py builds every weight from one width-independent
plan, so the masters are compatible by construction.  See that module's
docstring for the details.

Requirements:  pip install -r requirements.txt
               FontForge   (only to refresh skeleton.json)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SFD = os.path.join(HERE, "gortondigital.sfd")
SKELETON = os.path.join(HERE, "skeleton.json")
FEATURES = os.path.join(HERE, "features.fea")
DESIGNSPACE = os.path.join(HERE, "GortonDigital.designspace")
OUT_TTF = os.path.join(HERE, "GortonDigital-VF.ttf")
OUT_OTF = os.path.join(HERE, "GortonDigital-VF.otf")

FAMILY = "Gorton Digital"
# (style name, wght, pen width) -- pen widths are the same ones generateStrokes.ff uses
MASTERS = [
    ("Light", 300, 210),
    ("Regular", 400, 280),
    ("Medium", 500, 315),
    ("Heavy", 700, 420),
]
DEFAULT_MASTER = "Regular"
RIGHT_BEARING = 630  # same constant as generateStrokes.ff: advance = skeleton + width + 630
NO_STROKE = ("space", "uni00A0")  # generateStrokes.ff leaves these untouched

CAP_HEIGHT = 2520
X_HEIGHT = 1680

# vertical metrics.  The static fonts carry corrupt hhea/win values (descent
# of 10000+ units), so these are the intended values: typo metrics are the
# ones from the .sfd, win metrics are wide enough for the heaviest descenders.
TYPO_ASCENDER = 2940
TYPO_DESCENDER = -584
LINE_GAP = 420
WIN_ASCENT = 3500
WIN_DESCENT = 1250


def master_path(style):
    return os.path.join(HERE, "gortondigital%s.ufo" % style)


# --------------------------------------------------------------------------
# step 1: skeleton
# --------------------------------------------------------------------------
def ensure_skeleton():
    stale = (not os.path.exists(SKELETON)) or os.path.getmtime(SFD) > os.path.getmtime(SKELETON)
    if not stale:
        return
    fontforge = shutil.which("fontforge")
    if not fontforge:
        if os.path.exists(SKELETON):
            print("warning: gortondigital.sfd is newer than skeleton.json but FontForge is not installed; using skeleton.json")
            return
        sys.exit("error: skeleton.json is missing and FontForge is not installed")
    print("Exporting skeleton from gortondigital.sfd ...")
    subprocess.run(
        [fontforge, "-lang=py", "-script", os.path.join(HERE, "export_skeleton.py"), SFD, SKELETON],
        check=True,
        cwd=HERE,
    )


# --------------------------------------------------------------------------
# step 2: masters
# --------------------------------------------------------------------------
def notdef_skeleton(cap_height):
    """A stroked box for .notdef, so it matches the weight of the font."""
    w = 1200
    return {
        "unicode": None,
        "width": w,
        "contours": [
            {
                "closed": True,
                "points": [[0, 0, True], [0, cap_height, True], [w, cap_height, True], [w, 0, True]],
            }
        ],
    }


def make_pen_draw(contours, ndigits=2):
    """Replay rendered contours into a fontTools pen."""

    def r(pt):
        return (round(pt[0], ndigits), round(pt[1], ndigits))

    def draw(pen):
        for contour in contours:
            for seg in contour:
                if seg[0] == "M":
                    pen.moveTo(r(seg[1]))
                elif seg[0] == "L":
                    pen.lineTo(r(seg[1]))
                else:
                    pen.curveTo(r(seg[1]), r(seg[2]), r(seg[3]))
            pen.closePath()

    return draw


def build_masters():
    import ufoLib2

    import gorton_stroke as gs

    with open(SKELETON) as fh:
        skel = json.load(fh)

    glyphs = dict(skel["glyphs"])
    order = list(skel["order"])
    if ".notdef" not in glyphs:
        glyphs[".notdef"] = notdef_skeleton(CAP_HEIGHT)
        order.insert(0, ".notdef")

    r_values = [w / 2.0 for _, _, w in MASTERS]
    r_min, r_max = min(r_values), max(r_values)

    # one width-independent plan per glyph, shared by every master
    plans = {}
    for name in order:
        g = glyphs[name]
        if name in NO_STROKE or not g["contours"]:
            plans[name] = None
        else:
            plans[name] = gs.plan_glyph(g["contours"], r_min, r_max)

    with open(FEATURES) as fh:
        features = fh.read()

    for style, wght, width in MASTERS:
        r = width / 2.0
        path = master_path(style)
        font = ufoLib2.Font()
        info = font.info
        info.familyName = FAMILY
        info.styleName = style
        info.unitsPerEm = skel["unitsPerEm"]
        info.ascender = TYPO_ASCENDER
        info.descender = TYPO_DESCENDER
        info.capHeight = CAP_HEIGHT
        info.xHeight = X_HEIGHT
        info.italicAngle = 0
        info.versionMajor = 2017
        info.versionMinor = 10
        info.copyright = "This digital file is copyright 2017 R. S. Bartgis."
        info.openTypeOS2VendorID = "PfEd"
        info.openTypeOS2WeightClass = wght
        info.openTypeOS2WidthClass = 5
        info.openTypeOS2Type = []
        info.openTypeOS2Selection = [7]  # use typo metrics
        info.openTypeOS2TypoAscender = TYPO_ASCENDER
        info.openTypeOS2TypoDescender = TYPO_DESCENDER
        info.openTypeOS2TypoLineGap = LINE_GAP
        info.openTypeOS2WinAscent = WIN_ASCENT
        info.openTypeOS2WinDescent = WIN_DESCENT
        info.openTypeHheaAscender = TYPO_ASCENDER
        info.openTypeHheaDescender = TYPO_DESCENDER
        info.openTypeHheaLineGap = LINE_GAP
        info.postscriptUnderlinePosition = -420
        info.postscriptUnderlineThickness = width
        info.postscriptIsFixedPitch = False
        info.postscriptWeightName = style
        info.note = "Generated by build_variable.py from gortondigital.sfd -- do not edit by hand."
        font.features.text = features

        for name in order:
            g = glyphs[name]
            glyph = font.newGlyph(name)
            if g["unicode"]:
                glyph.unicodes = [g["unicode"]]
            plan = plans[name]
            if plan is None:
                glyph.width = g["width"]
                continue
            # same metrics as generateStrokes.ff: shift the skeleton right by the pen
            # width and widen the advance by the pen width plus RIGHT_BEARING
            glyph.width = g["width"] + width + RIGHT_BEARING
            contours = gs.render_plan(plan, r, dx=width)
            make_pen_draw(contours)(glyph.getPen())

        font.lib["public.glyphOrder"] = order
        if os.path.exists(path):
            shutil.rmtree(path)
        font.save(path)
        print("Wrote %s" % os.path.relpath(path, HERE))


# --------------------------------------------------------------------------
# step 3: variable font
# --------------------------------------------------------------------------
def fontmake_cmd():
    exe = shutil.which("fontmake")
    if exe:
        return [exe]
    return [sys.executable, "-m", "fontmake"]


def flag_overlaps(path):
    """
    Set OVERLAP_SIMPLE on every glyph of a TrueType variable font.

    The masters are made of deliberately overlapping contours.  Apple's
    rasterizer only fills those correctly when the first point of the glyph
    carries the OVERLAP_SIMPLE flag (other rasterizers ignore it).
    """
    from fontTools.ttLib import TTFont
    from fontTools.ttLib.tables._g_l_y_f import flagOverlapSimple

    font = TTFont(path)
    glyf = font["glyf"]
    count = 0
    for name in font.getGlyphOrder():
        glyph = glyf[name]
        if glyph.numberOfContours > 0:
            glyph.getCoordinates(glyf)  # decodes the flags
            glyph.flags[0] |= flagOverlapSimple
            count += 1
    font.save(path)
    print("Set OVERLAP_SIMPLE on %d glyphs" % count)


def build_variable():
    for out, fmt in ((OUT_TTF, "variable"), (OUT_OTF, "variable-cff2")):
        # --keep-overlaps: the overlapping contours ARE the outline (see gorton_stroke.py)
        cmd = fontmake_cmd() + ["-m", DESIGNSPACE, "-o", fmt, "--keep-overlaps", "--output-path", out]
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=HERE)
        if fmt == "variable":
            flag_overlaps(out)
        print("Wrote %s" % os.path.relpath(out, HERE))


def check_depth():
    """Refuse to build outlines that CoreText (macOS/iOS) would draw with holes."""
    try:
        import check_overlap_depth
    except ImportError as e:  # numpy / Pillow missing
        print("warning: skipping the overlap-depth check (%s)" % e)
        return
    if check_overlap_depth.main([]) != 0:
        sys.exit(
            "Too many overlapping contours: CoreText wraps its winding count at 16 layers.\n"
            "Simplify the skeleton or the stroker (see check_overlap_depth.py)."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masters", action="store_true", help="only regenerate the UFO masters")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    ensure_skeleton()
    check_depth()
    build_masters()
    if not args.masters:
        build_variable()


if __name__ == "__main__":
    main()
