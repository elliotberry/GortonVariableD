#!/usr/bin/env python3
"""
Interpolation-compatible stroker for the Gorton Digital skeleton.

Gorton is a monoline face: every glyph is a centre-line "skeleton" stroked with
a round pen (round caps, round joins).  FontForge's ExpandStroke produces good
static fonts, but it cleans up overlaps internally, so the outline structure
(point counts, contour counts, point types) changes from one stroke width to
the next.  Masters like that cannot be interpolated into a variable font.

This module strokes the skeleton with a deterministic construction instead.  It
works in two steps:

1. plan_glyph()  -- decide, from the skeleton ALONE, how the stroke is built:
                    which pieces are offset-curve "bodies", circular "pies",
                    straight "rects" and round "discs".  Nothing in the plan
                    depends on the pen width.
2. render_plan() -- turn that plan into contours for a given pen radius.

Because every master is rendered from the same plan, every master has exactly
the same contours, point counts, point types and start points.  The coordinates
of every primitive are affine in the pen radius, so intermediate weights
interpolate exactly.

The result is deliberately a set of OVERLAPPING contours (all wound the same
direction).  Their union is the true stroke; font renderers fill overlaps with
the non-zero winding rule, and variable fonts are allowed to contain overlaps.

Primitives (all closed, all counter-clockwise):

  rect   straight segment: a rectangle of half-width r.
  body   a run of smooth cubic pieces: right offset forward, left offset back.
         Only used where the curvature radius is comfortably larger than the
         heaviest pen radius, so the inner offset can never invert.
  pie    a circular arc whose radius is no bigger than the lightest pen
         radius: the stroke is exactly a circular sector of radius (rho + r).
  dot    a whole circle that small: a single disc of radius (rho + r).
  fan    a circular arc whose radius lies between the lightest pen radius and
         the "safe" limit: an annular sector with inner radius max(rho - r, 1).
         Exact whenever the pen is narrower or wider than the arc.
  ring   the same for a whole circle (outer circle plus a clockwise hole).
  disc   a circle of radius r: round caps and round joins at corners.
  wedge  a small polygon filling the outer side of a bend between two straight
         pieces (a round join without a full circle).
  Tight, non-circular curves (hooks, comma tails ...) are flattened into a
  polyline; each polyline segment is a rect and each joint a wedge, which
  approximates the Minkowski sum of the polyline with the pen to well under a
  unit.

Overlap depth
-------------
CoreText's rasterizer counts winding in four bits, so wherever 16 or more
contours stack up the count wraps to zero: holes, and hairlines running to the
edge of the glyph.  That is why joints are wedges rather than discs (a chain of
discs stacks up as deep as the pen is wide), why circular arcs use fans, and why
near-duplicate shapes are merged.  check_overlap_depth.py verifies every glyph
at every master stays comfortably below the limit, and build_variable.py runs it.
"""
import math

KAPPA = 0.5522847498307936  # cubic approximation of a quarter circle

# Geometry tolerances (font units; the em is 3524 units).
OFFSET_TOL = 0.35  # max error of an approximated offset curve
FLATTEN_TOL = 1.5  # max chord error when flattening tight curves (fewer, longer pieces = less overlap)
ARC_TOL = 0.6  # circle-fit tolerance when recognising circular arcs
CORNER_COS = math.cos(math.radians(0.5))  # tangents closer than this are smooth
SAFE_FACTOR = 1.15  # body only if curvature radius >= SAFE_FACTOR * r_max
MAX_PIECES = 16


# --------------------------------------------------------------------------
# small vector / bezier helpers
# --------------------------------------------------------------------------
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def _len(v):
    return math.hypot(v[0], v[1])


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _unit(v):
    n = _len(v)
    if n < 1e-9:
        return None
    return (v[0] / n, v[1] / n)


def bez(P, t):
    mt = 1.0 - t
    a, b, c, d = mt * mt * mt, 3 * mt * mt * t, 3 * mt * t * t, t * t * t
    return (
        a * P[0][0] + b * P[1][0] + c * P[2][0] + d * P[3][0],
        a * P[0][1] + b * P[1][1] + c * P[2][1] + d * P[3][1],
    )


def bez_d1(P, t):
    mt = 1.0 - t
    return tuple(
        3 * mt * mt * (P[1][i] - P[0][i])
        + 6 * mt * t * (P[2][i] - P[1][i])
        + 3 * t * t * (P[3][i] - P[2][i])
        for i in (0, 1)
    )


def bez_d2(P, t):
    mt = 1.0 - t
    return tuple(
        6 * mt * (P[2][i] - 2 * P[1][i] + P[0][i])
        + 6 * t * (P[3][i] - 2 * P[2][i] + P[1][i])
        for i in (0, 1)
    )


def sub_cubic(P, a, b):
    """The part of cubic P between parameters a and b, as a cubic."""
    h = (b - a) / 3.0
    p0, p3 = bez(P, a), bez(P, b)
    d0, d1 = bez_d1(P, a), bez_d1(P, b)
    return (
        p0,
        (p0[0] + h * d0[0], p0[1] + h * d0[1]),
        (p3[0] - h * d1[0], p3[1] - h * d1[1]),
        p3,
    )


def min_curvature_radius(P, n=48):
    """Smallest radius of curvature along cubic P (0 if the speed vanishes)."""
    best = math.inf
    for i in range(n + 1):
        t = i / n
        d1, d2 = bez_d1(P, t), bez_d2(P, t)
        speed = _len(d1)
        if speed < 1e-9:
            return 0.0
        k = abs(_cross(d1, d2)) / speed ** 3
        if k > 1e-12:
            best = min(best, 1.0 / k)
    return best


def total_turning(P, n=64):
    """Sum of absolute tangent-direction changes along cubic P, in radians."""
    prev, total = None, 0.0
    for i in range(n + 1):
        d1 = bez_d1(P, i / n)
        if _len(d1) < 1e-9:
            continue
        ang = math.atan2(d1[1], d1[0])
        if prev is not None:
            delta = (ang - prev + math.pi) % (2 * math.pi) - math.pi
            total += abs(delta)
        prev = ang
    return total


def offset_cubic(P, d):
    """
    Approximate the offset of cubic P by signed distance d (positive = left).

    End points and end tangents are exact; the handles are scaled by (1-d*k)
    (k = signed curvature), which is exactly the derivative of the true
    offset curve at the ends.  The result is affine in d.
    """
    ends = []
    for t in (0.0, 1.0):
        D, A = bez_d1(P, t), bez_d2(P, t)
        s = _len(D)
        normal = (-D[1] / s, D[0] / s)
        kappa = _cross(D, A) / s ** 3
        ends.append((D, normal, kappa))
    (d0, n0, k0), (d1, n1, k1) = ends
    o0 = (P[0][0] + d * n0[0], P[0][1] + d * n0[1])
    o3 = (P[3][0] + d * n1[0], P[3][1] + d * n1[1])
    f0, f1 = 1.0 - d * k0, 1.0 - d * k1
    c1 = (o0[0] + f0 * d0[0] / 3.0, o0[1] + f0 * d0[1] / 3.0)
    c2 = (o3[0] - f1 * d1[0] / 3.0, o3[1] - f1 * d1[1] / 3.0)
    return (o0, c1, c2, o3)


def _dist_to_polyline(pt, poly):
    best = math.inf
    for i in range(len(poly) - 1):
        a, b = poly[i], poly[i + 1]
        ab = _sub(b, a)
        denom = _dot(ab, ab)
        if denom < 1e-12:
            dist = _len(_sub(pt, a))
        else:
            u = max(0.0, min(1.0, _dot(_sub(pt, a), ab) / denom))
            dist = _len(_sub(pt, (a[0] + u * ab[0], a[1] + u * ab[1])))
        best = min(best, dist)
    return best


def offset_error(P, d, samples=24, dense=160):
    """Max deviation of the approximated offset curve from true distance |d|."""
    poly = [bez(P, i / dense) for i in range(dense + 1)]
    O = offset_cubic(P, d)
    err = 0.0
    for i in range(samples + 1):
        q = bez(O, i / samples)
        err = max(err, abs(_dist_to_polyline(q, poly) - abs(d)))
    return err


# --------------------------------------------------------------------------
# skeleton -> segments
# --------------------------------------------------------------------------
class Seg:
    """One line or cubic of a skeleton contour, between two on-curve points."""

    __slots__ = ("kind", "P", "v0", "v1", "degenerate")

    def __init__(self, kind, P, v0, v1):
        self.kind = kind  # 'line' or 'cubic'
        self.P = P  # 2 points (line) or 4 points (cubic)
        self.v0 = v0  # index of start vertex in the contour
        self.v1 = v1  # index of end vertex
        if kind == "line":
            self.degenerate = _len(_sub(P[1], P[0])) < 0.01
        else:
            self.degenerate = all(_len(_sub(p, P[0])) < 0.01 for p in P[1:])

    def tangent_start(self):
        P = self.P
        for p in P[1:]:
            u = _unit(_sub(p, P[0]))
            if u:
                return u
        return None

    def tangent_end(self):
        P = self.P
        for p in reversed(P[:-1]):
            u = _unit(_sub(P[-1], p))
            if u:
                return u
        return None


def is_straight(P, tol=1.5):
    """
    True if cubic P traces a straight line and never doubles back: both control
    points lie on the chord and between its ends.  FontForge skeletons often
    encode straight edges this way (e.g. P0 == P1), which would otherwise look
    like a zero-radius curve.
    """
    chord = _sub(P[3], P[0])
    length = _len(chord)
    if length < 0.01:
        return False
    u = (chord[0] / length, chord[1] / length)
    for p in (P[1], P[2]):
        rel = _sub(p, P[0])
        along = _dot(rel, u)
        if abs(_cross(u, rel)) > tol or along < -tol or along > length + tol:
            return False
    return True


def contour_segments(contour):
    """-> (vertices, segments, closed) for one skeleton contour."""
    pts = [(float(x), float(y), bool(on)) for x, y, on in contour["points"]]
    closed = bool(contour["closed"])
    on_idx = [i for i, p in enumerate(pts) if p[2]]
    if not on_idx:
        return [], [], closed
    # rotate so the contour starts on an on-curve point
    pts = pts[on_idx[0]:] + pts[: on_idx[0]]
    on_idx = [i for i, p in enumerate(pts) if p[2]]
    verts = [(pts[i][0], pts[i][1]) for i in on_idx]
    nseg = len(on_idx) if closed else len(on_idx) - 1
    segs = []
    for k in range(nseg):
        i = on_idx[k]
        j = on_idx[k + 1] if k + 1 < len(on_idx) else len(pts)
        offs = [(p[0], p[1]) for p in pts[i + 1:j]]
        a = verts[k]
        b = verts[(k + 1) % len(verts)]
        if len(offs) == 0:
            segs.append(Seg("line", (a, b), k, (k + 1) % len(verts)))
        elif len(offs) == 2:
            P = (a, offs[0], offs[1], b)
            if is_straight(P):
                segs.append(Seg("line", (a, b), k, (k + 1) % len(verts)))
            else:
                segs.append(Seg("cubic", P, k, (k + 1) % len(verts)))
        else:
            raise ValueError("skeleton must use cubic splines (got %d off-curve points)" % len(offs))
    return verts, segs, closed


# --------------------------------------------------------------------------
# circular arc recognition
# --------------------------------------------------------------------------
def _circle_from_3(p, q, r):
    ax, ay = p
    bx, by = q
    cx, cy = r
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-9:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    return (ux, uy), math.hypot(ax - ux, ay - uy)


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def fit_arc(P):
    """If cubic P is a circular arc: (centre, radius, start_angle, signed_sweep)."""
    fit = _circle_from_3(bez(P, 0.0), bez(P, 0.5), bez(P, 1.0))
    if fit is None:
        return None
    centre, rho = fit
    if rho > 1e5:
        return None
    for t in (0.125, 0.25, 0.375, 0.625, 0.75, 0.875):
        q = bez(P, t)
        if abs(math.hypot(q[0] - centre[0], q[1] - centre[1]) - rho) > ARC_TOL + 0.004 * rho:
            return None
    a0 = math.atan2(P[0][1] - centre[1], P[0][0] - centre[0])
    am = math.atan2(bez(P, 0.5)[1] - centre[1], bez(P, 0.5)[0] - centre[0])
    a1 = math.atan2(P[3][1] - centre[1], P[3][0] - centre[0])
    sweep = _wrap(am - a0) + _wrap(a1 - am)
    if abs(sweep) < 1e-6:
        return None
    return centre, rho, a0, sweep


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
class Plan:
    """An ordered list of primitives, independent of the pen width."""

    # Shapes closer than this (font units) count as duplicates.  Near-coincident
    # edges (a unit or two apart) also upset CoreText's scanline conversion, so
    # this is deliberately larger than any visible difference.
    DUP_TOL = 2.5

    def __init__(self):
        self.ops = []
        self._discs = {}  # key -> position (ordered), emitted at the end
        self._rects = []
        self._wedges = []

    def rect(self, a, b):
        """A straight piece; skips exact retraces (skeletons double back on themselves)."""
        t = self.DUP_TOL
        for c, d in self._rects:
            same = _len(_sub(a, c)) < t and _len(_sub(b, d)) < t
            flipped = _len(_sub(a, d)) < t and _len(_sub(b, c)) < t
            if same or flipped:
                return
        self._rects.append((a, b))
        self.ops.append(("rect", a, b))

    def disc(self, key, pos):
        self._discs.setdefault(key, pos)

    def wedge(self, pos, u1, u2):
        """Round join between two collinear-ish rects, as a small polygon.

        Using a whole disc at every joint of a flattened curve stacks dozens of
        circles on top of each other, and CoreText's rasterizer wraps its winding
        count at 16 layers (producing holes and stray hairlines).  A wedge only
        fills the gap on the outer side of the bend.
        """
        t = self.DUP_TOL
        for p, a, b in self._wedges:
            if _len(_sub(pos, p)) < t and abs(_cross(u1, a)) < 0.05:
                return
        self._wedges.append((pos, u1, u2))
        self.ops.append(("wedge", pos, u1, u2))

    def finish(self):
        placed = []
        for pos in self._discs.values():
            if any(_len(_sub(pos, q)) < self.DUP_TOL for q in placed):
                continue
            placed.append(pos)
            self.ops.append(("disc", pos))
        return self.ops


def _plan_curve(plan, ci, si, seg, r_min, r_max):
    """Plan one non-circular cubic: safe pieces become bodies, tight ones flatten."""
    P = seg.P
    rho_safe = SAFE_FACTOR * r_max
    m = max(1, int(math.ceil(total_turning(P) / math.radians(35.0))))
    pieces = None
    while True:
        pieces = []
        need_more = False
        for i in range(m):
            sub = sub_cubic(P, i / m, (i + 1) / m)
            safe = min_curvature_radius(sub) >= rho_safe
            if safe and m < MAX_PIECES:
                err = max(offset_error(sub, r_max), offset_error(sub, -r_max))
                if err > OFFSET_TOL:
                    need_more = True
            elif safe:
                if max(offset_error(sub, r_max), offset_error(sub, -r_max)) > OFFSET_TOL:
                    safe = False
            pieces.append((sub, safe))
        if need_more:
            m += 1
            continue
        break

    def boundary_key(j):
        if j == 0:
            return ("v", ci, seg.v0)
        if j == m:
            return ("v", ci, seg.v1)
        return ("b", ci, si, j)

    def join(key, pos, u1, u2):
        """Round join at a smooth joint: a wedge if the two sides are known and
        nearly parallel, otherwise a full disc."""
        if u1 is not None and u2 is not None and _dot(u1, u2) > 0.2:
            if abs(_cross(u1, u2)) > 1e-9:
                plan.wedge(pos, u1, u2)
            return
        plan.disc(key, pos)

    run = []
    # per piece: (is_body, direction on entering, direction on leaving)
    dirs = []
    for j, (sub, safe) in enumerate(pieces):
        if safe:
            run.append(sub)
            dirs.append((True, _unit(bez_d1(sub, 0.0)), _unit(bez_d1(sub, 1.0))))
            continue
        if run:
            plan.ops.append(("body", run))
            run = []
        # tight piece: flatten
        D = max(_len(bez_d2(sub, 0.0)), _len(bez_d2(sub, 1.0)))
        n = max(1, min(40, int(math.ceil(math.sqrt(D / (8.0 * FLATTEN_TOL))))))
        pts = [bez(sub, i / n) for i in range(n + 1)]
        solid = [_len(_sub(pts[i + 1], pts[i])) > 6.0 for i in range(n)]
        for i in range(n):
            if solid[i]:
                plan.rect(pts[i], pts[i + 1])
        for i in range(1, n):
            join(
                ("t", ci, si, j, i),
                pts[i],
                _unit(_sub(pts[i], pts[i - 1])) if solid[i - 1] else None,
                _unit(_sub(pts[i + 1], pts[i])) if solid[i] else None,
            )
        d_in = _unit(_sub(pts[1], pts[0])) if solid[0] else None
        d_out = _unit(_sub(pts[n], pts[n - 1])) if solid[n - 1] else None
        dirs.append((False, d_in, d_out))
        if j == 0:
            plan.disc(boundary_key(0), pts[0])
        if j == m - 1:
            plan.disc(boundary_key(m), pts[n])
    if run:
        plan.ops.append(("body", run))

    # joints between neighbouring pieces inside this curve
    for j in range(1, m):
        (body_a, _, out_a), (body_b, in_b, _) = dirs[j - 1], dirs[j]
        if body_a and body_b:
            continue  # one continuous body
        join(boundary_key(j), bez(P, j / m), out_a, in_b)


def plan_contour(plan, ci, contour, r_min, r_max):
    verts, segs, closed = contour_segments(contour)
    if not verts:
        return
    nv = len(verts)

    # A whole circle that is no bigger than the lightest pen: one disc.
    if closed and len(segs) >= 3 and all(s.kind == "cubic" for s in segs):
        centre = (sum(v[0] for v in verts) / nv, sum(v[1] for v in verts) / nv)
        rho = sum(math.hypot(v[0] - centre[0], v[1] - centre[1]) for v in verts) / nv
        if rho <= SAFE_FACTOR * r_max:
            arcs = [fit_arc(s.P) for s in segs]
            if all(a is not None for a in arcs) and all(
                abs(a[1] - rho) < 1.5 and math.hypot(a[0][0] - centre[0], a[0][1] - centre[1]) < 1.5
                for a in arcs
            ):
                if rho <= r_min + 1.0:
                    plan.ops.append(("dot", centre, rho))
                else:
                    plan.ops.append(("ring", centre, rho))
                return

    # classify segments
    kinds = []  # per segment: ('line',) ('arc', centre, rho, a0, sweep) ('curve',) ('skip',)
    for s in segs:
        if s.degenerate:
            kinds.append(("skip",))
        elif s.kind == "line":
            kinds.append(("line",))
        else:
            arc = fit_arc(s.P)
            if arc is not None and arc[1] <= SAFE_FACTOR * r_max:
                kinds.append(("arc",) + arc)
            else:
                kinds.append(("curve",))

    # merge neighbouring arcs that share a circle into one pie
    def same_circle(a, b):
        return (
            abs(a[2] - b[2]) < 1.0
            and math.hypot(a[1][0] - b[1][0], a[1][1] - b[1][1]) < 1.5
            and (a[4] > 0) == (b[4] > 0)
        )

    n = len(segs)
    start = 0
    if closed and n > 1:
        for i in range(n):  # start at a boundary between different items
            prev, cur = kinds[i - 1], kinds[i]
            if not (cur[0] == "arc" and prev[0] == "arc" and same_circle(prev, cur)):
                start = i
                break
    order = [(start + i) % n for i in range(n)]

    items = []  # (kind, [segment indices])
    for idx in order:
        k = kinds[idx]
        if (
            items
            and k[0] == "arc"
            and items[-1][0] == "arc"
            and same_circle(kinds[items[-1][1][-1]], k)
            and (closed or idx == items[-1][1][-1] + 1)
        ):
            items[-1][1].append(idx)
        else:
            items.append((k[0], [idx]))

    for kind, idxs in items:
        if kind == "skip":
            continue
        if kind == "line":
            s = segs[idxs[0]]
            plan.rect(s.P[0], s.P[1])
        elif kind == "arc":
            first, last = kinds[idxs[0]], kinds[idxs[-1]]
            sweep = sum(kinds[i][4] for i in idxs)
            small = first[2] <= r_min + 1.0  # no wider than the lightest pen
            if abs(sweep) > 2 * math.pi - 0.02:
                # a whole circle (e.g. the head of a comma): a pie would contain a
                # zero-width slit that some rasterizers draw as a hairline
                plan.ops.append(("dot" if small else "ring", first[1], first[2]))
            else:
                plan.ops.append(("pie" if small else "fan", first[1], first[2], first[3], sweep))
            plan.disc(("v", ci, segs[idxs[0]].v0), segs[idxs[0]].P[0])
            plan.disc(("v", ci, segs[idxs[-1]].v1), segs[idxs[-1]].P[-1])
        else:
            si = idxs[0]
            _plan_curve(plan, ci, si, segs[si], r_min, r_max)

    # round caps on open ends, round joins on corners
    for k in range(nv):
        if not closed and k in (0, nv - 1):
            plan.disc(("v", ci, k), verts[k])
            continue
        prev_i = (k - 1) % n if closed else k - 1
        next_i = k % n
        if not closed and not (0 <= prev_i < n and next_i < n):
            continue
        prev_s, next_s = segs[prev_i], segs[next_i]
        if prev_s.degenerate or next_s.degenerate:
            plan.disc(("v", ci, k), verts[k])
            continue
        u, v = prev_s.tangent_end(), next_s.tangent_start()
        if u is None or v is None or _dot(u, v) < CORNER_COS:
            plan.disc(("v", ci, k), verts[k])


def plan_glyph(contours, r_min, r_max):
    """Build the width-independent plan for one glyph's skeleton contours."""
    plan = Plan()
    for ci, contour in enumerate(contours):
        plan_contour(plan, ci, contour, r_min, r_max)
    return plan.finish()


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
# A rendered contour is a list of tuples: ('M', pt) ('L', pt) ('C', c1, c2, pt)
# and always closes implicitly.  Orientation is counter-clockwise (UFO).
def _circle(cx, cy, r):
    k = KAPPA * r
    return [
        ("M", (cx + r, cy)),
        ("C", (cx + r, cy + k), (cx + k, cy + r), (cx, cy + r)),
        ("C", (cx - k, cy + r), (cx - r, cy + k), (cx - r, cy)),
        ("C", (cx - r, cy - k), (cx - k, cy - r), (cx, cy - r)),
        ("C", (cx + k, cy - r), (cx + r, cy - k), (cx + r, cy)),
    ]


MIN_INNER = 1.0  # smallest inner radius of a fan / ring (see _fan)


def _arc_cubics(centre, a_from, a_to, rad, step_h):
    """One counter-clockwise (a_from < a_to) or clockwise quarter-or-less arc."""
    cx, cy = centre
    p = (cx + rad * math.cos(a_from), cy + rad * math.sin(a_from))
    q = (cx + rad * math.cos(a_to), cy + rad * math.sin(a_to))
    h = step_h * rad
    c1 = (p[0] - h * math.sin(a_from), p[1] + h * math.cos(a_from))
    c2 = (q[0] + h * math.sin(a_to), q[1] - h * math.cos(a_to))
    return p, c1, c2, q


def _fan(centre, rho, a0, sweep, r):
    """Stroke of a circular arc whose radius is between the pen radii.

    An annular sector from radius rho - r to rho + r.  Once the pen is wider than
    the arc (r >= rho) the true stroke is a plain sector from the arc's centre, so
    the inner radius is clamped at a small positive value instead of going
    negative (which would flip the inner arc through the centre and leave a hole
    where it overlaps neighbouring shapes).  Interpolating between masters that
    straddle rho is then only approximate, by a few units at most.
    """
    outer = rho + r
    inner = max(rho - r, MIN_INNER)
    lo, span = (a0, sweep) if sweep > 0 else (a0 + sweep, -sweep)
    n = max(1, int(math.ceil(span / (math.pi / 2) - 1e-9)))
    step = span / n
    hk = 4.0 / 3.0 * math.tan(step / 4.0)
    cx, cy = centre
    out = []
    for i in range(n):
        p, c1, c2, q = _arc_cubics(centre, lo + i * step, lo + (i + 1) * step, outer, hk)
        if i == 0:
            out.append(("M", p))
        out.append(("C", c1, c2, q))
    out.append(("L", (cx + inner * math.cos(lo + span), cy + inner * math.sin(lo + span))))
    for i in reversed(range(n)):
        a, b = lo + i * step, lo + (i + 1) * step
        p, c1, c2, q = _arc_cubics(centre, a, b, inner, hk)
        out.append(("C", c2, c1, p))  # same arc, walked backwards
    return out


def _ring(centre, rho, r):
    """Stroke of a whole circle between the pen radii: outer circle + hole."""
    cx, cy = centre
    outer = _circle(cx, cy, rho + r)
    s = max(rho - r, MIN_INNER)
    k = KAPPA * s
    hole = [  # clockwise
        ("M", (cx + s, cy)),
        ("C", (cx + s, cy - k), (cx + k, cy - s), (cx, cy - s)),
        ("C", (cx - k, cy - s), (cx - s, cy - k), (cx - s, cy)),
        ("C", (cx - s, cy + k), (cx - k, cy + s), (cx, cy + s)),
        ("C", (cx + k, cy + s), (cx + s, cy + k), (cx + s, cy)),
    ]
    return [outer, hole]


def _pie(centre, rho, a0, sweep, r):
    R = rho + r
    lo, span = (a0, sweep) if sweep > 0 else (a0 + sweep, -sweep)
    n = max(1, int(math.ceil(span / (math.pi / 2) - 1e-9)))
    step = span / n
    h = 4.0 / 3.0 * math.tan(step / 4.0) * R
    cx, cy = centre

    def at(a, rad=R):
        return (cx + rad * math.cos(a), cy + rad * math.sin(a))

    out = [("M", centre), ("L", at(lo))]
    for i in range(n):
        a, b = lo + i * step, lo + (i + 1) * step
        p, q = at(a), at(b)
        c1 = (p[0] - h * math.sin(a), p[1] + h * math.cos(a))
        c2 = (q[0] + h * math.sin(b), q[1] - h * math.cos(b))
        out.append(("C", c1, c2, q))
    return out


def render_plan(ops, r, dx=0.0):
    """Render a plan with pen radius r, shifted right by dx.  -> list of contours."""
    contours = []

    def sh(p):
        return (p[0] + dx, p[1])

    for op in ops:
        kind = op[0]
        if kind == "rect":
            a, b = op[1], op[2]
            u = _unit(_sub(b, a))
            n = (-u[1] * r, u[0] * r)
            contours.append(
                [
                    ("M", sh((a[0] - n[0], a[1] - n[1]))),
                    ("L", sh((b[0] - n[0], b[1] - n[1]))),
                    ("L", sh((b[0] + n[0], b[1] + n[1]))),
                    ("L", sh((a[0] + n[0], a[1] + n[1]))),
                ]
            )
        elif kind == "body":
            right = [offset_cubic(p, -r) for p in op[1]]
            left = [offset_cubic(p, r) for p in op[1]]
            c = [("M", sh(right[0][0]))]
            for o in right:
                c.append(("C", sh(o[1]), sh(o[2]), sh(o[3])))
            c.append(("L", sh(left[-1][3])))
            for o in reversed(left):
                c.append(("C", sh(o[2]), sh(o[1]), sh(o[0])))
            contours.append(c)
        elif kind == "wedge":
            _, p, u1, u2 = op
            left_turn = _cross(u1, u2) > 0
            if left_turn:  # the outer side of the bend is the right-hand side
                n1, n2 = (u1[1], -u1[0]), (u2[1], -u2[0])
            else:
                n1, n2 = (-u1[1], u1[0]), (-u2[1], u2[0])
            bis = _unit((n1[0] + n2[0], n1[1] + n2[1]))
            pts = [
                p,
                (p[0] + r * n1[0], p[1] + r * n1[1]),
                (p[0] + r * bis[0], p[1] + r * bis[1]),
                (p[0] + r * n2[0], p[1] + r * n2[1]),
            ]
            if not left_turn:
                pts = [pts[0], pts[3], pts[2], pts[1]]
            contours.append([("M", sh(pts[0]))] + [("L", sh(q)) for q in pts[1:]])
        elif kind == "disc":
            contours.append(_circle(op[1][0] + dx, op[1][1], r))
        elif kind == "dot":
            contours.append(_circle(op[1][0] + dx, op[1][1], op[2] + r))
        elif kind == "ring":
            _, centre, rho = op
            contours.extend(_ring((centre[0] + dx, centre[1]), rho, r))
        elif kind == "fan":
            _, centre, rho, a0, sweep = op
            contours.append(_fan((centre[0] + dx, centre[1]), rho, a0, sweep, r))
        elif kind == "pie":
            _, centre, rho, a0, sweep = op
            c = _pie((centre[0] + dx, centre[1]), rho, a0, sweep, r)
            contours.append(c)
        else:
            raise ValueError(kind)
    return contours


def contour_area(contour):
    """Signed area of a rendered contour (flattened); > 0 means counter-clockwise."""
    pts = []
    cur = None
    for seg in contour:
        if seg[0] in ("M", "L"):
            cur = seg[1]
            pts.append(cur)
        else:
            c1, c2, p = seg[1], seg[2], seg[3]
            for i in range(1, 9):
                pts.append(bez((cur, c1, c2, p), i / 8.0))
            cur = p
    area = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        area += x0 * y1 - x1 * y0
    return area / 2.0
