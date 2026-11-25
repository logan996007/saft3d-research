"""Float64 geometric predicates for the experimental advancing-front mesher.

All ``tol`` arguments are *absolute distances* in the input coordinate system.
The mesher normalizes its entire domain to a unit bounding box before calling
these functions.  Penetrations shallower than ``tol`` are treated as contact.
These are floating-point predicates, not adaptive/exact arithmetic: callers
must reject degenerate or poorly conditioned candidates using volume and
quality thresholds.  No guarantee is made for nearly degenerate configurations.
"""

from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np


_EDGES = tuple(combinations(range(4), 2))
_EDGE_IDS = np.asarray(_EDGES, dtype=np.intp)
_FACE_IDS = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.intp)
_NORMAL_EPS = 32.0 * np.finfo(np.float64).eps


def _points(value: object, count: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (count, 3) or not np.isfinite(result).all():
        raise ValueError(f"expected {count} finite 3D points")
    return result


def _tolerance(tol: float) -> float:
    value = float(tol)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("tol must be a finite nonnegative absolute distance")
    return value


def signed_volume(tet: object) -> float:
    """Return det(b-a, c-a, d-a)/6; positive means the accepted orientation."""
    vertices = _points(tet, 4)
    ab, ac, ad = vertices[1:] - vertices[0]
    return float(np.dot(ab, np.cross(ac, ad)) / 6.0)


def tet_quality(tet: object) -> float:
    """Normalized mean ratio: 1 for a regular tet, 0 if inverted/degenerate.

    For positive volume V this is 12*(3*V)**(2/3)/sum(edge_length**2).
    A local scaling avoids unnecessary overflow/underflow in the quality ratio.
    """
    vertices = _points(tet, 4)
    shifted = vertices - vertices[0]
    scale = float(np.max(np.abs(shifted)))
    if scale == 0.0:
        return 0.0
    shifted = shifted / scale
    determinant = float(np.dot(shifted[1], np.cross(shifted[2], shifted[3])))
    if determinant <= 0.0:
        return 0.0
    edge_sum = sum(float(np.dot(shifted[j] - shifted[i], shifted[j] - shifted[i]))
                   for i, j in _EDGES)
    quality = 12.0 * (determinant / 2.0) ** (2.0 / 3.0) / edge_sum
    return float(np.clip(quality, 0.0, 1.0))


def tet_faces(ids4: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
    """Outward faces of a positive tet, in opposite-vertex order 0, 1, 2, 3.

    This is purely combinatorial; it cannot infer or repair an inverted tet.
    """
    if len(ids4) != 4 or len(set(ids4)) != 4:
        raise ValueError("a tetrahedron requires four distinct vertex IDs")
    a, b, c, d = (int(value) for value in ids4)
    return ((b, c, d), (a, d, c), (a, b, d), (a, c, b))


def _outward_planes(vertices: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Unit outward normals and plane anchors; [] for a degenerate tet."""
    ab, ac, ad = vertices[1:] - vertices[0]
    determinant = float(np.dot(ab, np.cross(ac, ad)))
    if determinant == 0.0:
        return []
    sign = 1.0 if determinant > 0.0 else -1.0
    planes = []
    for i, j, k in tet_faces((0, 1, 2, 3)):
        u = vertices[j] - vertices[i]
        v = vertices[k] - vertices[i]
        normal = np.cross(u, v)
        length = float(np.linalg.norm(normal))
        if length == 0.0:
            return []
        planes.append((sign * normal / length, vertices[i]))
    return planes


def point_in_tet(p: object, tet: object, tol: float = 1e-10) -> str:
    """Classify a point as ``inside``, ``boundary``, or ``outside``.

    Either tet orientation is supported.  A zero-volume tet has no interior and
    returns ``outside``.  The boundary band extends by tol on both sides.
    """
    tolerance = _tolerance(tol)
    vertices = _points(tet, 4)
    point = np.asarray(p, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("expected a finite 3D point")
    # Translating first reduces cancellation for coordinates far from the origin.
    point = point - vertices[0]
    vertices = vertices - vertices[0]
    planes = _outward_planes(vertices)
    if not planes:
        return "outside"
    distances = np.array([np.dot(normal, point - anchor) for normal, anchor in planes])
    if np.any(distances > tolerance):
        return "outside"
    return "inside" if np.all(distances < -tolerance) else "boundary"


def tet_overlap(a: object, b: object, tol: float = 1e-10) -> bool:
    """Return whether two tets have interior overlap exceeding the tolerance.

    Separating-axis testing uses both sets of four face normals and all 36
    cross products of edges.  Shared faces, edges, or vertices are allowed.
    Either orientation is accepted.  Zero-volume inputs return False and must
    be rejected separately by callers; this function does not validate tets.
    """
    tolerance = _tolerance(tol)
    first, second = _points(a, 4), _points(b, 4)
    origin = first[0].copy()
    first, second = first - origin, second - origin
    ab, ac, ad = first[1:] - first[0]
    if float(np.dot(ab, np.cross(ac, ad))) == 0.0:
        return False
    ab, ac, ad = second[1:] - second[0]
    if float(np.dot(ab, np.cross(ac, ad))) == 0.0:
        return False

    def separated(axes):
        # The rows are unit axes, so overlap depth has distance units.
        projection_a, projection_b = first @ axes.T, second @ axes.T
        depth = np.minimum(projection_a.max(axis=0), projection_b.max(axis=0)) - np.maximum(
            projection_a.min(axis=0), projection_b.min(axis=0))
        return bool(np.any(depth <= tolerance))

    face_vertices = np.concatenate((first[_FACE_IDS], second[_FACE_IDS]), axis=0)
    normals = np.cross(face_vertices[:, 1] - face_vertices[:, 0],
                       face_vertices[:, 2] - face_vertices[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths == 0.0):
        return False
    # Most neighboring front cells are separated by a face plane.  Avoid the
    # edge-axis allocation for those common cases.
    if separated(normals / lengths[:, None]):
        return False
    edges_a = first[_EDGE_IDS[:, 1]] - first[_EDGE_IDS[:, 0]]
    edges_b = second[_EDGE_IDS[:, 1]] - second[_EDGE_IDS[:, 0]]
    axes = np.cross(edges_a[:, None, :], edges_b[None, :, :]).reshape(-1, 3)
    lengths = np.linalg.norm(axes, axis=1)
    thresholds = _NORMAL_EPS * np.outer(np.linalg.norm(edges_a, axis=1),
                                        np.linalg.norm(edges_b, axis=1)).ravel()
    # Parallel edges contribute no separating axis.  Omitted nearly parallel
    # axes can produce a conservative false-positive collision.
    keep = lengths > thresholds
    return not np.any(keep) or not separated(axes[keep] / lengths[keep, None])


def triangle_intersects_tet_interior(
    tri: object, tet: object, tol: float = 1e-10
) -> bool:
    """Test whether positive triangle area enters a tet's strict interior.

    Clip the triangle against the four inward-offset (by tol) tetrahedron
    halfspaces.  Original boundary triangles coplanar with tet faces therefore
    remain legal.  This does not detect nonconforming contacts on the boundary;
    topology and boundary-coverage checks belong in a separate validator.
    """
    tolerance = _tolerance(tol)
    triangle, vertices = _points(tri, 3), _points(tet, 4)
    origin = vertices[0].copy()
    triangle, vertices = triangle - origin, vertices - origin
    planes = _outward_planes(vertices)
    if not planes:
        return False
    polygon = [point.copy() for point in triangle]
    for normal, anchor in planes:
        if not polygon:
            return False
        clipped: list[np.ndarray] = []
        previous = polygon[-1]
        previous_distance = float(np.dot(normal, previous - anchor)) + tolerance
        for current in polygon:
            current_distance = float(np.dot(normal, current - anchor)) + tolerance
            previous_inside = previous_distance < 0.0
            current_inside = current_distance < 0.0
            if previous_inside != current_inside:
                fraction = previous_distance / (previous_distance - current_distance)
                clipped.append(previous + fraction * (current - previous))
            if current_inside:
                clipped.append(current)
            previous, previous_distance = current, current_distance
        polygon = clipped
    if len(polygon) < 3:
        return False
    # A fan triangulation of this convex planar polygon has consistent winding.
    area_vector = np.zeros(3, dtype=np.float64)
    for index in range(1, len(polygon) - 1):
        area_vector += np.cross(polygon[index] - polygon[0], polygon[index + 1] - polygon[0])
    return bool(float(np.linalg.norm(area_vector)) > 2.0 * tolerance * tolerance)
