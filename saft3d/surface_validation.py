"""Small-surface embeddedness checks, independent of tetrahedron predicates.

The caller supplies coordinates normalized to a unit-size domain.  This module
complements the mesher's edge/orientation/connectivity/volume checks: it rejects
coincident vertex IDs, non-manifold vertex links and geometrical intersections
outside a shared indexed simplex.  HiGHS uses floating-point tolerances; these
checks are not an exact-arithmetic proof of embeddedness.
"""

from collections import defaultdict

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import cKDTree


_MAX_FACES = 2048
_LP_OPTIONS = {
    "primal_feasibility_tolerance": 1.0e-10,
    "dual_feasibility_tolerance": 1.0e-10,
    "ipm_optimality_tolerance": 1.0e-12,
}


def _check_vertex_links(faces):
    links = defaultdict(list)
    for a, b, c in faces:
        a, b, c = int(a), int(b), int(c)
        links[a].append((b, c))
        links[b].append((c, a))
        links[c].append((a, b))
    for vertex, edges in links.items():
        adjacency = defaultdict(list)
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        if len(adjacency) < 3 or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise ValueError(
                f"surface vertex {vertex} has a non-manifold link: "
                "every link vertex must have degree two in a single cycle"
            )
        seen = set()
        pending = [next(iter(adjacency))]
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            seen.add(current)
            pending.extend(v for v in adjacency[current] if v not in seen)
        if len(seen) != len(adjacency):
            raise ValueError(
                f"surface vertex {vertex} is non-manifold: "
                "its link contains more than one disconnected cycle"
            )


def _one_side(distances, tol):
    return bool(np.all(distances > tol) or np.all(distances < -tol))


def _check_triangle_pair(a, b, ids_a, ids_b, normal_a, normal_b, i, j, tol):
    shared = set(map(int, ids_a)).intersection(map(int, ids_b))
    exclusive_a = np.array([int(v) not in shared for v in ids_a])
    exclusive_b = np.array([int(v) not in shared for v in ids_b])

    # If all non-shared vertices are strictly on one side of the other plane,
    # this triangle can meet that plane only in its shared indexed simplex.
    # This includes the safe fast path for non-coplanar triangles sharing an edge.
    dist_a_to_b = (a - b[0]) @ normal_b
    dist_b_to_a = (b - a[0]) @ normal_a
    plane_tol = max(tol, 64.0 * np.finfo(float).eps)
    if (_one_side(dist_a_to_b[exclusive_a], plane_tol)
            or _one_side(dist_b_to_a[exclusive_b], plane_tol)):
        return

    # Two equal convex combinations describe triangle-triangle intersection.
    # For adjacent triangles, maximize barycentric mass outside their common
    # vertex IDs.  A zero optimum means all intersections lie on the shared
    # vertex or edge; positive mass detects crossings and coplanar overlap.
    combined = np.concatenate((a, b))
    origin = combined.min(axis=0)
    scale = float(np.max(np.ptp(combined, axis=0)))
    a_local, b_local = (a - origin) / scale, (b - origin) / scale
    equality = np.zeros((5, 6))
    equality[0, :3] = 1.0
    equality[1, 3:] = 1.0
    equality[2:, :3] = a_local.T
    equality[2:, 3:] = -b_local.T
    objective = -np.concatenate((exclusive_a, exclusive_b)).astype(float) if shared else np.zeros(6)
    try:
        result = linprog(
            objective,
            A_eq=equality,
            b_eq=np.array([1.0, 1.0, 0.0, 0.0, 0.0]),
            bounds=(0.0, None),
            method="highs",
            options=_LP_OPTIONS,
        )
    except Exception as exc:
        raise ValueError(
            f"surface intersection LP failed for triangles {i} and {j}: {exc}"
        ) from exc
    if result.status == 2 and not shared:
        return  # A certified numerical infeasibility result, within LP tolerance.
    if not result.success:
        raise ValueError(
            f"surface intersection LP failed for triangles {i} and {j}: "
            f"status {result.status}, {result.message}"
        )
    solution = np.asarray(result.x, dtype=float)
    residual = equality @ solution - np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    if (not np.isfinite(solution).all() or not np.isfinite(result.fun)
            or np.min(solution) < -1.0e-9 or np.max(np.abs(residual)) > 1.0e-9):
        raise ValueError(
            f"surface intersection LP failed for triangles {i} and {j}: "
            "returned an invalid numerical feasibility certificate"
        )
    # HiGHS feasibility resolution imposes a minimum barycentric tolerance;
    # the distance tolerance is also converted to the local pair's scale.
    bary_tol = max(1.0e-9, 8.0 * tol / scale)
    if not shared or -float(result.fun) > bary_tol:
        if shared:
            simplex = "vertex" if len(shared) == 1 else "edge"
            raise ValueError(
                f"surface triangles {i} and {j} have a self-intersection or "
                f"nonconforming contact outside their shared indexed {simplex}"
            )
        raise ValueError(
            f"surface triangles {i} and {j} self-intersect or touch "
            "without a shared vertex index (nonconforming contact)"
        )


def check_embedded_surface(vertices, faces, tol=1.0e-10):
    """Return None for a numerically embedded closed 2-manifold surface.

    Raise ``ValueError`` with the offending vertex/triangle IDs on failure.
    Intended to follow the mesher's existing surface checks; at most 2048
    triangles are accepted.  All triangle pairs are considered, with AABB and
    plane-side rejection before independent barycentric intersection LPs.
    Solver failures fail closed.  ``tol`` is a distance in normalized units.
    """
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("surface tolerance must be finite and nonnegative")
    try:
        points = np.asarray(vertices, dtype=float)
        raw_faces = np.asarray(faces)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("surface vertices/faces must be rectangular numeric arrays") from exc
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all()):
        raise ValueError("surface vertices must be a nonempty finite Nx3 array")
    if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces):
        raise ValueError("surface faces must be a nonempty Fx3 array")
    if len(raw_faces) > _MAX_FACES:
        raise ValueError(
            f"embedded-surface validation is limited to {_MAX_FACES} triangles; "
            f"received {len(raw_faces)}"
        )
    try:
        numeric_faces = np.asarray(raw_faces, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("surface faces must contain integer vertex indices") from exc
    if (not np.isfinite(numeric_faces).all()
            or not np.equal(numeric_faces, np.floor(numeric_faces)).all()
            or np.any(numeric_faces < 0) or np.any(numeric_faces >= len(points))):
        raise ValueError("surface faces contain invalid or out-of-range vertex indices")
    triangles = numeric_faces.astype(np.int64)
    face_keys = set()
    for i, face in enumerate(triangles):
        key = tuple(sorted(map(int, face)))
        if len(set(key)) != 3:
            raise ValueError(f"surface triangle {i} repeats a vertex index")
        if key in face_keys:
            raise ValueError(f"surface triangle {i} duplicates another indexed triangle")
        face_keys.add(key)

    coincident = cKDTree(points).query_pairs(tol, output_type="ndarray")
    if len(coincident):
        i, j = map(int, coincident[0])
        distance = float(np.linalg.norm(points[i] - points[j]))
        raise ValueError(
            f"surface vertices {i} and {j} are duplicate or near-coincident "
            f"(distance {distance:.6g} <= tolerance {tol:.6g}); use a single vertex index"
        )

    xyz = points[triangles]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    edge_scale = np.maximum.reduce([
        np.linalg.norm(xyz[:, 1] - xyz[:, 0], axis=1),
        np.linalg.norm(xyz[:, 2] - xyz[:, 0], axis=1),
        np.linalg.norm(xyz[:, 2] - xyz[:, 1], axis=1),
    ])
    area_tol = np.maximum(tol * edge_scale, 64 * np.finfo(float).eps * edge_scale**2)
    bad = np.flatnonzero(lengths <= area_tol)
    if len(bad):
        raise ValueError(f"surface triangle {int(bad[0])} is degenerate within tolerance")
    normals /= lengths[:, None]
    _check_vertex_links(triangles)

    lows, highs = xyz.min(axis=1), xyz.max(axis=1)
    for i in range(len(triangles) - 1):
        nearby = np.flatnonzero(np.all(
            (highs[i] + tol >= lows[i + 1:])
            & (highs[i + 1:] + tol >= lows[i]), axis=1,
        )) + i + 1
        for j in nearby:
            j = int(j)
            _check_triangle_pair(
                xyz[i], xyz[j], triangles[i], triangles[j],
                normals[i], normals[j], i, j, tol,
            )
    return None
