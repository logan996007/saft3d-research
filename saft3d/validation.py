"""Independent, tolerance-based checks for small tetrahedral meshes.

The generator's intersection code is deliberately not reused here.  Candidate
tetrahedron pairs use conservative separating-plane certificates, then
SciPy/HiGHS linear programs in barycentric coordinates for unresolved pairs,
including contacts outside a shared indexed simplex.  These are
floating-point numerical checks, not an exact-arithmetic mathematical proof.
Boundary triangles must be oriented outward and tetrahedra positively oriented.
"""

from collections import Counter, defaultdict
from itertools import combinations
from time import perf_counter

import numpy as np
from scipy.optimize import linprog


_BARY_TOL = 1.0e-8
_DETAIL_LIMIT = 50
_LP_OPTIONS = {
    "primal_feasibility_tolerance": 1.0e-9,
    "dual_feasibility_tolerance": 1.0e-9,
    "ipm_optimality_tolerance": 1.0e-10,
}
# Outward boundary of a positively oriented tetrahedron [0, 1, 2, 3].
_OUTWARD_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
_EDGES = tuple(combinations(range(4), 2))
_FACE_INDEX = np.asarray(_OUTWARD_FACES)
# Certificates require a margin comfortably beyond the LP feasibility band.
# These constants apply after each pair is normalized to unit coordinate extent.
_CERT_MARGIN = 64.0 * _BARY_TOL
_CERT_ROUNDOFF = 128.0 * np.finfo(float).eps


def _face_orientation(face):
    """Sign of a triangle's permutation relative to its sorted vertex key."""
    inversions = sum(face[i] > face[j] for i in range(3) for j in range(i + 1, 3))
    return -1 if inversions % 2 else 1


def _plane_separation_kind(a, b, ids_a, ids_b):
    """Return a conservative ``disjoint``/``conforming`` certificate, or None.

    This does not use the generator's SAT predicates. For disjoint indexed
    cells, face-normal projections must have a *strict positive gap*. For
    shared vertices, a plane through their simplex must put every other vertex
    strictly on opposite sides. Convexity then confines the whole intersection
    to that indexed simplex. Mere touching, nearly coplanar nonshared vertices,
    and all uncertain cases receive no certificate and still go through LPs.

    To find planes through a shared edge/vertex, we combine extreme rays of a
    tiny homogeneous halfspace cone. The candidate construction itself need not
    be reliable: a separate final vertex-side check authorizes the certificate.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ids_a, ids_b = tuple(map(int, ids_a)), tuple(map(int, ids_b))
    if len(set(ids_a)) != 4 or len(set(ids_b)) != 4:
        return None
    shared = set(ids_a).intersection(ids_b)
    if len(shared) == 4:
        return None  # Identical indexed cells overlap, rather than merely touch.
    combined = np.concatenate((a, b))
    extent = float(np.max(np.ptp(combined, axis=0)))
    if not np.isfinite(extent) or extent <= 0:
        return None
    if not shared:
        origin = combined.min(axis=0)
        first, second = (a - origin) / extent, (b - origin) / extent
        triangles = np.concatenate((first[_FACE_INDEX], second[_FACE_INDEX]))
        normals = np.cross(triangles[:, 1] - triangles[:, 0],
                           triangles[:, 2] - triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        usable = lengths > _CERT_ROUNDOFF
        if not usable.any():
            return None
        normals = normals[usable] / lengths[usable, None]
        projection_a, projection_b = first @ normals.T, second @ normals.T
        gaps = np.maximum(projection_b.min(axis=0) - projection_a.max(axis=0),
                          projection_a.min(axis=0) - projection_b.max(axis=0))
        return "disjoint" if np.any(gaps > _CERT_MARGIN) else None

    shared_ids = sorted(shared)
    shared_a = np.array([a[ids_a.index(vertex)] for vertex in shared_ids])
    shared_b = np.array([b[ids_b.index(vertex)] for vertex in shared_ids])
    # The real mesh shares one global coordinate array. Do not certify an
    # inconsistent direct call that merely labels different points identically.
    if not np.array_equal(shared_a, shared_b):
        return None
    origin = shared_a[0]
    first, second = (a - origin) / extent, (b - origin) / extent
    common = (shared_a - origin) / extent
    exclusive_a = first[[vertex not in shared for vertex in ids_a]]
    exclusive_b = second[[vertex not in shared for vertex in ids_b]]

    def certify(normal):
        length = float(np.linalg.norm(normal))
        if not np.isfinite(length) or length <= _CERT_ROUNDOFF:
            return False
        normal = normal / length
        if np.max(np.abs(common @ normal)) > _CERT_ROUNDOFF:
            return False
        da, db = exclusive_a @ normal, exclusive_b @ normal
        return bool((np.all(da < -_CERT_MARGIN) and np.all(db > _CERT_MARGIN))
                    or (np.all(db < -_CERT_MARGIN) and np.all(da > _CERT_MARGIN)))

    if len(shared) == 3:
        normal = np.cross(common[1], common[2])
        return "conforming" if certify(normal) else None

    # Seek n with n dot (-a_exclusive) > 0 and n dot b_exclusive > 0.
    # For a common edge, also require n perpendicular to that edge.
    directions = np.concatenate((-exclusive_a, exclusive_b))
    edge = None
    if len(shared) == 2:
        edge = common[1]
        edge_len = float(np.linalg.norm(edge))
        if edge_len <= _CERT_ROUNDOFF:
            return None
        edge = edge / edge_len
        directions = directions - (directions @ edge)[:, None] * edge
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths <= _CERT_ROUNDOFF):
        return None
    directions = directions / lengths[:, None]
    if certify(directions.sum(axis=0)):
        return "conforming"

    if edge is not None:
        rays = np.cross(edge, directions)
    else:
        pairs = np.asarray(list(combinations(range(len(directions)), 2)))
        rays = np.cross(directions[pairs[:, 0]], directions[pairs[:, 1]])
    ray_lengths = np.linalg.norm(rays, axis=1)
    usable = ray_lengths > _CERT_ROUNDOFF
    if not usable.any():
        return None
    rays = rays[usable] / ray_lengths[usable, None]
    rays = np.concatenate((rays, -rays))
    feasible = rays[np.all(directions @ rays.T >= -_CERT_ROUNDOFF, axis=0)]
    if len(feasible) and certify(feasible.sum(axis=0)):
        return "conforming"
    return None


def _intersection_with_certificate(a, b, ids_a, ids_b):
    """Return (kind, actual LP calls, certified), preserving the original LPs."""
    kind = _plane_separation_kind(a, b, ids_a, ids_b)
    if kind is not None:
        return kind, 0, True
    kind, calls = _intersection_kind(a, b, ids_a, ids_b)
    return kind, calls, False


def _intersection_kind(a, b, ids_a, ids_b):
    """Return (kind, LP calls): disjoint, conforming, overlap, bad contact, error.

    Equal convex combinations locate the intersection.  Maximizing a common
    lower bound on all eight barycentric weights detects intersection of the
    *interiors*, not just a shared face/edge/vertex.  A second LP checks that any
    boundary intersection is confined to the simplex of shared vertex IDs.
    """
    combined = np.concatenate((a, b))
    origin = combined.min(axis=0)
    scale = float(np.max(np.ptp(combined, axis=0)))
    if not scale > 0:
        return "LP error: zero pair extent", 0
    a = (a - origin) / scale
    b = (b - origin) / scale
    equality = np.zeros((5, 9), dtype=float)
    equality[0, :4] = 1.0
    equality[1, 4:8] = 1.0
    equality[2:, :4] = a.T
    equality[2:, 4:8] = -b.T
    rhs = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    inequality = np.zeros((8, 9), dtype=float)
    inequality[:, :8] = -np.eye(8)
    inequality[:, 8] = 1.0
    objective = np.zeros(9)
    objective[8] = -1.0
    result = linprog(
        objective,
        A_ub=inequality,
        b_ub=np.zeros(8),
        A_eq=equality,
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs",
        options=_LP_OPTIONS,
    )
    if result.status == 2:
        return "disjoint", 1
    if not result.success:
        return "LP error: " + str(result.message), 1
    if float(result.x[8]) > _BARY_TOL:
        return "overlap", 1

    shared = set(map(int, ids_a)).intersection(map(int, ids_b))
    if not shared:
        # A geometrical contact with no common indexed vertex is not conforming.
        return "nonconforming contact", 1
    exclusive = np.array(
        [int(v) not in shared for v in ids_a]
        + [int(v) not in shared for v in ids_b],
        dtype=float,
    )
    if not np.any(exclusive):
        return "conforming", 1  # Duplicated tets are reported separately.
    contact = linprog(
        -exclusive,
        A_eq=equality[:, :8],
        b_eq=rhs,
        bounds=(0.0, None),
        method="highs",
        options=_LP_OPTIONS,
    )
    if not contact.success:
        return "LP error: " + str(contact.message), 2
    if -float(contact.fun) > _BARY_TOL:
        return "nonconforming contact", 2
    return "conforming", 2


def _index_array(value, width, name, vertex_count, errors):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{name} must have shape (N, {width}).")
        return None
    if raw.size == 0:
        return np.empty((0, width), dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != width:
        errors.append(f"{name} must have shape (N, {width}).")
        return None
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError):
        errors.append(f"{name} must contain integer vertex indices.")
        return None
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        errors.append(f"{name} must contain finite integer vertex indices.")
        return None
    if np.any(numeric < 0) or np.any(numeric >= vertex_count):
        errors.append(f"{name} contains an out-of-range vertex index.")
        return None
    return numeric.astype(np.int64)


def validate_mesh(vertices, tetrahedra, boundary_faces, *, check_overlap=True):
    """Validate a small mesh and return a JSON-serializable diagnostics dict.

    ``vertices`` is N x 3, ``tetrahedra`` is M x 4, and ``boundary_faces`` is
    F x 3.  Exact face matching means equality of *vertex-index triples*, not
    approximate spatial proximity.  All tetrahedron pairs are enumerated when
    ``check_overlap`` is true; AABBs reject distant pairs before independent
    plane certificates and, for uncertified cases, the original LPs.
    ``checked_pairs`` counts pairs submitted to that geometric checker.
    ``certified_pairs`` is the subset resolved without an LP; ``lp_calls`` counts
    actual solver invocations, including two calls for difficult contact cases.

    Unused vertices are informational.  Setting ``check_overlap=False`` skips
    both overlap and nonconforming-contact checks and is stated in the report.
    ``valid`` only means that the enabled, tolerance-based checks passed.
    """
    start = perf_counter()
    errors = []
    report = {
        "valid": False,
        "errors": errors,
        "vertex_count": 0,
        "tet_count": 0,
        "boundary_face_count": 0,
        "extracted_boundary_face_count": 0,
        "checked_pairs": 0,
        "certified_pairs": 0,
        "total_pairs": 0,
        "lp_calls": 0,
        "overlap_count": 0,
        "overlap_pairs": [],
        "nonconforming_contact_count": 0,
        "nonconforming_contact_pairs": [],
        "intersection_failure_count": 0,
        "intersection_failures": [],
        "geometric_intersections_checked": bool(check_overlap),
        "numerical_validation": True,
        "validation_note": (
            "Floating-point, tolerance-based validation; not an exact-arithmetic "
            "proof. Geometric pair details are capped at 50; counts are exhaustive."
        ),
    }

    def finish():
        report["valid"] = not errors
        report["validation_seconds"] = float(perf_counter() - start)
        return report

    try:
        points = np.asarray(vertices, dtype=float)
    except (TypeError, ValueError, OverflowError):
        errors.append("vertices must be a finite numeric array of shape (N, 3).")
        return finish()
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        errors.append("vertices must be a finite numeric array of shape (N, 3).")
        return finish()
    report["vertex_count"] = len(points)
    tets = _index_array(tetrahedra, 4, "tetrahedra", len(points), errors)
    boundary = _index_array(boundary_faces, 3, "boundary_faces", len(points), errors)
    if tets is None or boundary is None:
        return finish()
    report["tet_count"] = len(tets)
    report["boundary_face_count"] = len(boundary)
    report["total_pairs"] = len(tets) * (len(tets) - 1) // 2
    if not len(tets):
        errors.append("The mesh contains no tetrahedra.")
    if not len(boundary):
        errors.append("The prescribed boundary contains no triangles.")
    if not len(points):
        return finish()

    referenced = np.unique(np.concatenate((tets.ravel(), boundary.ravel())))
    active_points = points[referenced] if len(referenced) else points
    scale = float(np.max(np.ptp(active_points, axis=0)))
    if not np.isfinite(scale) or scale <= 0.0:
        errors.append("The vertices have zero or non-finite spatial extent.")
        return finish()
    try:
        volume_scale = scale**3
    except OverflowError:
        volume_scale = float("inf")
    if not np.isfinite(volume_scale) or volume_scale < np.finfo(float).tiny:
        errors.append("Coordinate scale is outside the supported floating-point volume range.")
        return finish()
    volume_tol = 1.0e-12 * volume_scale
    coordinate_tol = 1.0e-10 * scale
    report["tolerances"] = {
        "volume": float(volume_tol),
        "coordinate": float(coordinate_tol),
        "barycentric": _BARY_TOL,
        "relative_volume": 1.0e-7,
    }
    repeated_tets = [i for i, tet in enumerate(tets) if len(set(tet)) != 4]
    repeated_boundary = [i for i, face in enumerate(boundary) if len(set(face)) != 3]
    if repeated_tets:
        errors.append(f"{len(repeated_tets)} tetrahedra repeat a vertex index.")
    if repeated_boundary:
        errors.append(f"{len(repeated_boundary)} boundary triangles repeat a vertex index.")

    coords = points[tets]
    if len(tets):
        matrices = np.stack(
            (coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0],
             coords[:, 3] - coords[:, 0]), axis=-1,
        )
        volumes = np.linalg.det(matrices) / 6.0
    else:
        volumes = np.empty(0)
    positive = volumes > volume_tol
    inverted = volumes < -volume_tol
    degenerate = np.abs(volumes) <= volume_tol
    report.update({
        "positive_tet_count": int(np.count_nonzero(positive)),
        "inverted_tet_count": int(np.count_nonzero(inverted)),
        "degenerate_tet_count": int(np.count_nonzero(degenerate)),
        "min_signed_volume": float(volumes.min()) if len(volumes) else None,
        "signed_volume_sum": float(volumes.sum()),
        "positive_volume_min": float(volumes[positive].min()) if np.any(positive) else None,
        "positive_volume_sum": float(volumes[positive].sum()),
    })
    if np.any(inverted):
        errors.append(f"{report['inverted_tet_count']} tetrahedra have negative signed volume.")
    if np.any(degenerate):
        errors.append(f"{report['degenerate_tet_count']} tetrahedra are degenerate within tolerance.")

    tet_keys = Counter(tuple(sorted(map(int, tet))) for tet in tets)
    duplicates = sum(count - 1 for count in tet_keys.values())
    report["duplicate_tet_count"] = duplicates
    if duplicates:
        errors.append(f"{duplicates} duplicate tetrahedra were found.")

    faces = defaultdict(list)
    for tet_id, tet in enumerate(tets):
        for local in _OUTWARD_FACES:
            face = tuple(int(tet[j]) for j in local)
            faces[tuple(sorted(face))].append((tet_id, _face_orientation(face)))
    extracted = {key: entries[0][1] for key, entries in faces.items() if len(entries) == 1}
    nonmanifold = [key for key, entries in faces.items() if len(entries) > 2]
    bad_internal = [
        key for key, entries in faces.items()
        if len(entries) == 2 and entries[0][1] == entries[1][1]
    ]
    report.update({
        "extracted_boundary_face_count": len(extracted),
        "internal_face_count": sum(len(entries) == 2 for entries in faces.values()),
        "nonmanifold_face_count": len(nonmanifold),
        "internal_orientation_error_count": len(bad_internal),
        "face_manifold": not nonmanifold,
        "internal_faces_oppositely_oriented": not bad_internal,
    })
    if nonmanifold:
        errors.append(f"{len(nonmanifold)} faces belong to more than two tetrahedra.")
    if bad_internal:
        errors.append(f"{len(bad_internal)} internal faces have equal rather than opposite orientations.")

    supplied = defaultdict(list)
    for face in boundary:
        face = tuple(map(int, face))
        supplied[tuple(sorted(face))].append(_face_orientation(face))
    duplicate_boundary = sum(len(entries) - 1 for entries in supplied.values())
    missing = sorted(set(extracted) - set(supplied))
    extra = sorted(set(supplied) - set(extracted))
    bad_boundary_orientation = [
        key for key in set(extracted).intersection(supplied)
        if len(supplied[key]) == 1 and supplied[key][0] != extracted[key]
    ]
    report.update({
        "boundary_faces_match": not missing and not extra and not duplicate_boundary,
        "missing_boundary_face_count": len(missing),
        "extra_boundary_face_count": len(extra),
        "missing_boundary_faces": [list(key) for key in missing[:_DETAIL_LIMIT]],
        "extra_boundary_faces": [list(key) for key in extra[:_DETAIL_LIMIT]],
        "duplicate_boundary_face_count": duplicate_boundary,
        "boundary_orientation_error_count": len(bad_boundary_orientation),
    })
    if missing or extra:
        errors.append(
            f"Boundary triangle keys differ: {len(missing)} mesh boundary faces are missing "
            f"from the prescribed boundary; {len(extra)} prescribed faces are absent from the mesh boundary."
        )
    if duplicate_boundary:
        errors.append(f"{duplicate_boundary} duplicate boundary triangles were found.")
    if bad_boundary_orientation:
        errors.append(f"{len(bad_boundary_orientation)} boundary triangles have incorrect orientation.")

    boundary_edges = defaultdict(list)
    for triangle in boundary:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(triangle[i]), int(triangle[j])
            boundary_edges[tuple(sorted((a, b)))].append(1 if a < b else -1)
    bad_boundary_edges = sum(
        len(signs) != 2 or sum(signs) != 0 for signs in boundary_edges.values()
    )
    report["boundary_edge_error_count"] = bad_boundary_edges
    report["boundary_closed_oriented"] = bool(boundary_edges) and not bad_boundary_edges
    if bad_boundary_edges:
        errors.append(f"{bad_boundary_edges} boundary edges are not incident to two oppositely oriented triangles.")

    # Translating before the divergence-theorem sum improves conditioning.
    shifted = points - active_points.mean(axis=0)
    boundary_xyz = shifted[boundary]
    boundary_volume = float(np.einsum(
        "ij,ij->i", boundary_xyz[:, 0],
        np.cross(boundary_xyz[:, 1], boundary_xyz[:, 2]),
    ).sum() / 6.0)
    relative_error = abs(float(volumes.sum()) - boundary_volume) / max(abs(boundary_volume), volume_tol)
    report["signed_boundary_volume"] = boundary_volume
    report["relative_volume_error"] = float(relative_error)
    if boundary_volume <= volume_tol:
        errors.append("The prescribed boundary has nonpositive or negligible signed volume.")
    if relative_error > 1.0e-7:
        errors.append(f"Mesh and boundary signed volumes differ (relative error {relative_error:.6g}).")

    used = set(map(int, tets.ravel()))
    unused = sorted(set(range(len(points))) - used)
    report["unused_vertex_count"] = len(unused)
    report["unused_vertex_indices"] = unused
    edge_sum = sum(np.sum((coords[:, i] - coords[:, j]) ** 2, axis=1) for i, j in _EDGES)
    quality = np.divide(
        12.0 * np.power(3.0 * np.abs(volumes), 2.0 / 3.0), edge_sum,
        out=np.zeros(len(tets)), where=edge_sum > 0,
    )
    report["quality"] = {
        "metric": "mean_ratio: 12*(3*abs(volume))^(2/3)/sum(squared_edge_lengths)",
        "ideal": 1.0,
        "min": float(np.min(quality)) if len(quality) else None,
        "p05": float(np.percentile(quality, 5)) if len(quality) else None,
        "median": float(np.median(quality)) if len(quality) else None,
    }

    if check_overlap and len(tets) > 1:
        lows, highs = coords.min(axis=1), coords.max(axis=1)
        for i in range(len(tets) - 1):
            if degenerate[i]:
                continue
            candidates = np.flatnonzero(np.all(
                (highs[i] + coordinate_tol >= lows[i + 1:])
                & (highs[i + 1:] + coordinate_tol >= lows[i]), axis=1,
            )) + i + 1
            for j in candidates:
                j = int(j)
                if degenerate[j]:
                    continue
                report["checked_pairs"] += 1
                kind, calls, certified = _intersection_with_certificate(coords[i], coords[j], tets[i], tets[j])
                report["certified_pairs"] += int(certified)
                report["lp_calls"] += calls
                if kind == "overlap":
                    report["overlap_count"] += 1
                    if len(report["overlap_pairs"]) < _DETAIL_LIMIT:
                        report["overlap_pairs"].append([i, j])
                elif kind == "nonconforming contact":
                    report["nonconforming_contact_count"] += 1
                    if len(report["nonconforming_contact_pairs"]) < _DETAIL_LIMIT:
                        report["nonconforming_contact_pairs"].append([i, j])
                elif kind.startswith("LP error"):
                    report["intersection_failure_count"] += 1
                    if len(report["intersection_failures"]) < _DETAIL_LIMIT:
                        report["intersection_failures"].append({"pair": [i, j], "reason": kind})
        if report["overlap_count"]:
            errors.append(f"{report['overlap_count']} tetrahedron pairs overlap with positive volume.")
        if report["nonconforming_contact_count"]:
            errors.append(f"{report['nonconforming_contact_count']} tetrahedron pairs have nonconforming contact.")
        if report["intersection_failure_count"]:
            errors.append(f"{report['intersection_failure_count']} pair intersection LPs failed.")
    return finish()
