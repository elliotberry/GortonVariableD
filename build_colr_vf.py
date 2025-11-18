#!/usr/bin/env python3
"""
Build a COLRv1 variable OTF (CFF host) with a wght axis using alpha crossfades
between Light/Regular/Medium/Heavy OTF masters.

Inputs:
  - gortondigitalLight.otf
  - gortondigitalRegular.otf
  - gortondigitalMedium.otf
  - gortondigitalHeavy.otf

Output:
  - GortonDigital-VF.otf

Notes:
- This uses COLRv1 PaintGlyph + PaintVarSolid layers with per-layer alpha that
  varies along the weight axis, achieving a visual variable stroke width effect
  without outline compatibility between masters.
- Requires: fonttools>=4.60
"""
import os
import sys
from typing import Dict, List, Tuple

from fontTools.ttLib import TTFont, newTable
from fontTools.ttLib.tables import otTables as ot
from fontTools.colorLib.builder import buildCPAL
from fontTools.varLib.varStore import OnlineVarStoreBuilder
from fontTools.ttLib.tables._f_v_a_r import table__f_v_a_r, Axis, NamedInstance


MASTERS: List[Tuple[str, int, str]] = [
    ("gortondigitalLight.otf", 300, "Light"),
    ("gortondigitalRegular.otf", 400, "Regular"),
    ("gortondigitalMedium.otf", 500, "Medium"),
    ("gortondigitalHeavy.otf", 700, "Heavy"),
]


def fail(msg: str) -> None:
    print(f"Error: {msg}")
    sys.exit(1)


def ensure_files(paths: List[str]) -> None:
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        fail("Missing files: " + ", ".join(missing))


def tent_alpha(x: float, center: float, left: float, right: float) -> float:
    if x <= left or x >= right:
        return 0.0
    if x == center:
        return 1.0
    if x < center:
        return (x - left) / (center - left)
    return (right - x) / (right - center)


def build_var_alpha_varstore(axis_tag: str, axis_min: int, axis_def: int, axis_max: int,
                             peaks: List[int]) -> Tuple[ot.VarStore, ot.DeltaSetIndexMap, Dict[int, int]]:
    # Build a VarStore with a single axis and produce varIdx for each master alpha envelope
    vs_builder = OnlineVarStoreBuilder([axis_tag])
    # Supports must be dicts of axis -> (start, peak, end)
    supports = [
        {axis_tag: (axis_min, axis_min, axis_min)},
        {axis_tag: (axis_def, axis_def, axis_def)},
        {axis_tag: (axis_max, axis_max, axis_max)},
    ]
    vs_builder.setSupports(supports)

    varidx_by_center: Dict[int, int] = {}
    for i, center in enumerate(peaks):
        left = peaks[i - 1] if i > 0 else axis_min
        right = peaks[i + 1] if i + 1 < len(peaks) else axis_max
        # Deltas at supports [left, center, right]
        deltas = [0.0, 1.0 if center == axis_def else 0.0, 0.0]
        varIdx = vs_builder.storeDeltas(deltas)
        varidx_by_center[center] = varIdx

    varStore = vs_builder.finish()
    # Build a trivial DeltaSetIndexMap that maps varIdx directly (base=0 outer=0 inner=varIdx)
    dim = ot.DeltaSetIndexMap()
    dim.Format = 0
    dim.MapCount = 0
    dim.EntryFormat = 0
    return varStore, dim, varidx_by_center


def main() -> None:
    ensure_files([m[0] for m in MASTERS])

    # Use Regular as base font to host tables
    base_path = MASTERS[1][0]
    font = TTFont(base_path)

    # Build CPAL palette (single black swatch)
    cpal = buildCPAL([[(0, 0, 0, 1.0)]])
    font["CPAL"] = cpal

    # Prepare VarStore for alpha
    axis_tag = "wght"
    axis_min, axis_def, axis_max = 300, 400, 700
    peaks = [w for _, w, _ in MASTERS]
    varStore, varIndexMap, varidx_by_center = build_var_alpha_varstore(axis_tag, axis_min, axis_def, axis_max, peaks)

    # Build COLR paints: for each base glyph, layer 4 PaintGlyphs with VarSolid alpha controlled by wght
    # Collect glyph sets from each master
    master_fonts = [(w, TTFont(p)) for p, w, _ in MASTERS]
    # Ensure glyph names are aligned (intersection)
    common_glyphs = None
    for _, tf in master_fonts:
        names = set(tf.getGlyphOrder())
        common_glyphs = names if common_glyphs is None else (common_glyphs & names)
    if not common_glyphs:
        fail("No common glyphs across masters")

    # Exclude .notdef and glyphs with zero width if desired
    exclude = {".notdef"}
    base_glyphs = sorted(common_glyphs - exclude)

    # Install master outlines into base OTF's CFF charstrings under suffixed names
    if 'CFF ' not in font:
        fail("Base OTF must have 'CFF ' table; use OTF masters.")
    from fontTools.pens.recordingPen import RecordingPen
    from fontTools.pens.t2CharStringPen import T2CharStringPen
    
    def build_t2_charstring_from_recording(rec_ops, glyphset, width):
        pen = T2CharStringPen(glyphSet=glyphset, width=width)
        path_open = False
        for op, pts in rec_ops:
            if op == 'moveTo':
                if path_open:
                    try:
                        pen.endPath()
                    except Exception:
                        pass
                if not pts:
                    continue
                pen.moveTo(pts[0])
                path_open = True
            elif op == 'lineTo':
                for pt in pts:
                    pen.lineTo(pt)
            elif op == 'curveTo':
                if len(pts) % 3 != 0:
                    continue
                for i in range(0, len(pts), 3):
                    pen.curveTo(pts[i], pts[i+1], pts[i+2])
            elif op in ('closePath', 'endPath'):
                try:
                    pen.endPath()
                except Exception:
                    pass
                path_open = False
            else:
                continue
        try:
            pen.endPath()
        except Exception:
            pass
        return pen.getCharString(private=top.Private, globalSubrs=top.GlobalSubrs)
    top = font['CFF '].cff.topDictIndex[0]
    cs_base = top.CharStrings
    base_glyph_order = font.getGlyphOrder()
    new_order = list(base_glyph_order)
    suffix_to_glyphset = {name: tf.getGlyphSet() for (w, tf), name in zip(master_fonts, [m[2] for m in MASTERS])}
    base_glyphset = font.getGlyphSet()
    for base_name in base_glyphs:
        for suffix, glyphset in suffix_to_glyphset.items():
            if base_name not in glyphset.keys():
                continue
            rec = RecordingPen()
            try:
                glyphset[base_name].draw(rec)
            except Exception:
                # Skip glyphs that fail to decompile/draw
                continue
            try:
                new_cs = build_t2_charstring_from_recording(rec.value, base_glyphset, glyphset[base_name].width)
            except Exception:
                continue
            new_name = f"{base_name}.{suffix}"
            if new_name in cs_base.charStrings:
                continue
            cs_base.charStrings[new_name] = new_cs
            if new_name not in top.charset:
                top.charset.append(new_name)
            adv = font['hmtx'].metrics.get(base_name, (glyphset[base_name].width, 0))[0]
            font['hmtx'].metrics[new_name] = (adv, 0)
            new_order.append(new_name)
    font.setGlyphOrder(new_order)

    # Validate and repair CFF charstrings: drop any layer glyphs that fail bounds/draw
    def validate_cff_charstrings():
        bad = []
        for gname in list(top.charset):
            if '.' not in gname:
                continue
            cs = cs_base.charStrings.get(gname)
            try:
                # Trigger bounds calculation to catch invalid programs
                if cs is not None:
                    cs.calcBounds(cs_base)
            except Exception:
                bad.append(gname)
        for gname in bad:
            try:
                # Remove from CFF and metrics/order
                del cs_base.charStrings[gname]
            except Exception:
                pass
            try:
                top.charset.remove(gname)
            except ValueError:
                pass
            font['hmtx'].metrics.pop(gname, None)
        if bad:
            font.setGlyphOrder([n for n in font.getGlyphOrder() if n not in set(bad)])
        return bad

    _bad = validate_cff_charstrings()

    # Construct LayerList and BaseGlyphList using otTables
    # Collect layer paints globally
    layerList = ot.LayerList()
    layerPaints: List[ot.Paint] = []

    # Helper: glyph name to GID
    glyphOrder = font.getGlyphOrder()
    gid_by_name = {name: i for i, name in enumerate(glyphOrder)}

    baseGlyphRecords: List[ot.BaseGlyphPaintRecord] = []
    for base_name in base_glyphs:
        first_layer_index = len(layerPaints)
        # add 1 paint per master
        for (_, wght, name) in MASTERS:
            varIdx = varidx_by_center[wght]
            # PaintVarSolid
            pvs = ot.Paint(); pvs.Format = ot.PaintFormat.PaintVarSolid
            pvs.PaletteIndex = 0
            pvs.Alpha = 0.0
            pvs.VarIndexBase = varIdx
            # PaintGlyph wrapping the suffixed glyph
            pg = ot.Paint(); pg.Format = ot.PaintFormat.PaintGlyph
            pg.Paint = pvs
            target_name = f"{base_name}.{name}"
            gid = gid_by_name.get(target_name)
            if gid is None:
                # layer glyph was removed (out-of-range), skip this layer
                continue
            pg.Glyph = gid
            layerPaints.append(pg)

        # PaintColrLayers referencing the contiguous block we just appended
        pcl = ot.Paint(); pcl.Format = ot.PaintFormat.PaintColrLayers
        pcl.NumLayers = len(layerPaints) - first_layer_index
        pcl.FirstLayerIndex = first_layer_index

        # BaseGlyphPaintRecord for this base glyph
        bpr = ot.BaseGlyphPaintRecord()
        base_gid = gid_by_name.get(base_name)
        if pcl.NumLayers > 0 and base_gid is not None:
            bpr.BaseGlyph = base_gid
            bpr.Paint = pcl
            baseGlyphRecords.append(bpr)

    # Finalize LayerList and BaseGlyphList
    layerList.LayerCount = len(layerPaints)
    layerList.Paint = layerPaints

    baseGlyphList = ot.BaseGlyphList()
    baseGlyphList.BaseGlyphCount = len(baseGlyphRecords)
    baseGlyphList.BaseGlyphPaintRecord = baseGlyphRecords

    # CFF host: no glyf bounds concerns

    # Build COLR table
    colrTable = newTable('COLR')
    colrTable.Version = 1
    # v0 fields empty
    colrTable.BaseGlyphRecordCount = 0
    colrTable.LayerRecordCount = 0
    # v1 fields
    colrTable.BaseGlyphList = baseGlyphList
    colrTable.LayerList = layerList
    colrTable.ClipList = None
    colrTable.VarIndexMap = varIndexMap
    colrTable.VarStore = varStore

    font['COLR'] = colrTable

    # Ensure fvar axis exists so variations apply
    if 'fvar' not in font:
        fvar = table__f_v_a_r()
        fvar.axes = []
        fvar.instances = []
        # Add axis name to 'name' table
        name = font['name']
        axisNameID = name.addName('Weight')
        axis = Axis()
        axis.axisTag = axis_tag
        axis.minValue = float(axis_min)
        axis.defaultValue = float(axis_def)
        axis.maxValue = float(axis_max)
        axis.flags = 0
        axis.axisNameID = axisNameID
        fvar.axes.append(axis)
        font['fvar'] = fvar

    out_path = "GortonDigital-VF.otf"
    if len(sys.argv) > 1:
        out_path = sys.argv[1]
    font.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
