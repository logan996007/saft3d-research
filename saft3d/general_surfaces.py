"""Sparse analytical surface fixtures outside the star-kernel restriction.

These surfaces are constructed directly from polygons and oriented box walls.
No tetrahedral volume mesh or volume meshing library is used to obtain them.
"""

from __future__ import annotations

import numpy as np


GENERAL_SURFACE_NAMES = ("u_prism", "hollow_cube", "slender_s")


def _signed_polygon_area(polygon):
    p = np.asarray(polygon, dtype=float)
    return float(np.sum(p[:, 0] * np.roll(p[:, 1], -1)
                        - p[:, 1] * np.roll(p[:, 0], -1)) / 2.0)


def _cross2(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _triangulate_polygon(polygon):
    """Deterministic ear clipping for these simple counterclockwise polygons.

    An ear is accepted only when no other active vertex lies inside or on it.
    Thus a cap diagonal cannot pass through an existing boundary vertex.
    This small helper is not a general polygon-repair routine.
    """
    p = np.asarray(polygon, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("polygon must contain at least three finite 2D vertices")
    scale = float(np.max(np.ptp(p, axis=0)))
    eps = 64 * np.finfo(float).eps * scale * scale
    if _signed_polygon_area(p) <= eps:
        raise ValueError("polygon must be counterclockwise and nondegenerate")
    active = list(range(len(p)))
    triangles = []
    while len(active) > 3:
        for index, b in enumerate(active):
            a, c = active[index - 1], active[(index + 1) % len(active)]
            if _cross2(p[b] - p[a], p[c] - p[a]) <= eps:
                continue
            blocked = False
            for v in active:
                if v in (a, b, c):
                    continue
                signs = (_cross2(p[b] - p[a], p[v] - p[a]),
                         _cross2(p[c] - p[b], p[v] - p[b]),
                         _cross2(p[a] - p[c], p[v] - p[c]))
                if min(signs) >= -eps:
                    blocked = True
                    break
            if not blocked:
                triangles.append((a, b, c))
                del active[index]
                break
        else:
            raise ValueError("no valid ear found; polygon is degenerate or not simple")
    a, b, c = active
    if _cross2(p[b] - p[a], p[c] - p[a]) <= eps:
        raise ValueError("degenerate final ear")
    triangles.append((a, b, c))
    return np.asarray(triangles, dtype=np.int64)


def _extrude_polygon(polygon, height):
    """Extrude a CCW polygon about z=0, returning its outward surface only."""
    p = np.asarray(polygon, dtype=float)
    if not np.isfinite(height) or height <= 0:
        raise ValueError("height must be finite and positive")
    n = len(p)
    cap = _triangulate_polygon(p)
    bottom = np.column_stack((p, np.full(n, -height / 2)))
    top = np.column_stack((p, np.full(n, height / 2)))
    faces = [tuple(face[::-1]) for face in cap]
    faces.extend(tuple(face + n) for face in cap)
    for a in range(n):
        b = (a + 1) % n
        faces.extend(((a, b, b + n), (a, b + n, a + n)))
    return np.vstack((bottom, top)), np.asarray(faces, dtype=np.int64)


def _box_surface(half_side):
    vertices = half_side * np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ], dtype=float)
    quads = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))
    faces = [(a, b, c) for a, b, c, d in quads]
    faces += [(a, c, d) for a, b, c, d in quads]
    return vertices, np.asarray(faces, dtype=np.int64)


def make_general_surface(name):
    """Return ``(vertices, outward_triangles, metadata)`` for a sparse fixture.

    ``hole_seeds`` identifies voids, not additional connected solids.  The
    hollow cube's inner shell points toward the cavity (out of the material).
    All fixtures deliberately have no positive-radius visibility kernel.
    """
    aliases = {"u": "u_prism", "u_shape": "u_prism", "hollow": "hollow_cube",
               "cavity": "hollow_cube", "s": "slender_s", "s_prism": "slender_s"}
    normalized = str(name).lower().strip().replace("-", "_")
    name = aliases.get(normalized, normalized)
    metadata = {"name": name, "expected_kernel": False, "hole_seeds": [],
                "boundary_components": 1}
    if name == "u_prism":
        polygon = np.array([(0, 0), (3, 0), (3, 3), (2, 3),
                            (2, 1), (1, 1), (1, 3), (0, 3)], dtype=float)
        vertices, faces = _extrude_polygon(polygon, height=1.0)
        metadata.update({
            "analytic_volume": 7.0,  # (3*3 - 1*2) * 1
            "description": "U-shaped prism with an open notch; its two arms have incompatible visibility half-spaces.",
            "polygon_vertices": len(polygon), "extrusion_height": 1.0,
        })
    elif name == "hollow_cube":
        outer_vertices, outer_faces = _box_surface(1.0)
        inner_vertices, inner_faces = _box_surface(0.4)
        vertices = np.vstack((outer_vertices, inner_vertices))
        faces = np.vstack((outer_faces, inner_faces[:, ::-1] + len(outer_vertices)))
        metadata.update({
            "analytic_volume": 8.0 - 0.8 ** 3,
            "hole_seeds": [[0.0, 0.0, 0.0]], "boundary_components": 2,
            "description": "Cube with a concentric cubic cavity; the inner shell has reversed material-outward orientation.",
            "outer_side": 2.0, "inner_side": 0.8,
        })
    elif name == "slender_s":
        # Three length-4 strips of width .3, joined by two .3 x .9 bridges.
        polygon = np.array([(0, 0), (4, 0), (4, .3), (.3, .3),
                            (.3, 1.2), (4, 1.2), (4, 2.7), (0, 2.7),
                            (0, 2.4), (3.7, 2.4), (3.7, 1.5), (0, 1.5)], dtype=float)
        vertices, faces = _extrude_polygon(polygon, height=.24)
        metadata.update({
            "analytic_volume": (3 * 4 * .3 + 2 * .3 * .9) * .24,
            "description": "Thin S-shaped prism with two opposite bends and narrow connecting passages.",
            "polygon_vertices": len(polygon), "extrusion_height": .24,
            "passage_width": .3,
        })
    else:
        raise ValueError(f"unknown general surface {name!r}; choose {', '.join(GENERAL_SURFACE_NAMES)}")
    return vertices, faces, metadata
