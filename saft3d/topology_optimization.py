"""Small-patch 2<->3 tetrahedral flips with fixed coordinates and boundary.

This is a bounded local quality optimizer, not a Delaunay mesher and not a
guarantee of eliminating all slivers.  Geometry checks use float64 tolerances.
The input must already be a conforming, non-overlapping volume mesh; a separate
validator should verify that precondition and the final output when necessary.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from time import perf_counter

import numpy as np

from .geometry import signed_volume, tet_faces, tet_overlap, tet_quality


_MAX_ACCEPTED_PER_PASS = 128
_MAX_ACCEPTED_TOTAL = 384
_DISTANCE_TOL = 1e-10
_VOLUME_TOL = 128 * np.finfo(float).eps


def _indices(value, width, count, name):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != width or not len(raw):
        raise ValueError(f"{name} must be a nonempty Nx{width} array")
    numeric = np.asarray(raw, dtype=float)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer indices")
    if np.any(numeric < 0) or np.any(numeric >= count):
        raise ValueError(f"{name} has an out-of-range index")
    result = numeric.astype(np.int64)
    if any(len(set(row)) != width for row in result):
        raise ValueError(f"{name} repeats an index within a simplex")
    return result


def _orientation(face):
    return -1 if sum(face[i] > face[j] for i in range(3) for j in range(i + 1, 3)) % 2 else 1


def _boundary_signature(cells):
    """Return oriented exterior keys, or None if a patch is nonmanifold."""
    incidence = defaultdict(list)
    for cell in cells:
        for face in tet_faces(cell):
            incidence[tuple(sorted(face))].append(_orientation(face))
    exterior = {}
    for face, signs in incidence.items():
        if len(signs) == 1:
            exterior[face] = signs[0]
        elif len(signs) != 2 or signs[0] == signs[1]:
            return None
    return exterior


def _adjacency(cells):
    faces, edges = defaultdict(list), defaultdict(list)
    for index, cell in enumerate(cells):
        for face in tet_faces(cell):
            faces[tuple(sorted(face))].append(index)
        for edge in combinations(cell, 2):
            edges[tuple(sorted(map(int, edge)))].append(index)
    return faces, edges


def _quality_summary(values):
    return {"min": float(np.min(values)), "p05": float(np.percentile(values, 5)),
            "median": float(np.median(values))}


def _candidate_patch(points, old_cells, proposed, min_gain):
    """Orient and certify one replacement in a unit-size local patch."""
    ids = np.unique(old_cells)
    origin = points[ids].min(axis=0)
    scale = float(np.max(np.ptp(points[ids], axis=0)))
    if not scale > 0.0:
        return None, "degenerate_patch"
    local = {int(i): (points[int(i)] - origin) / scale for i in ids}
    proposed = np.asarray(proposed, dtype=np.int64).copy()
    old_coords = [np.asarray([local[int(i)] for i in cell]) for cell in old_cells]
    new_coords = []
    for cell in proposed:
        if len(set(map(int, cell))) != 4:
            return None, "repeated_vertex"
        coords = np.asarray([local[int(i)] for i in cell])
        volume = signed_volume(coords)
        if abs(volume) <= _VOLUME_TOL:
            return None, "degenerate_new_tet"
        if volume < 0:
            cell[0], cell[1] = cell[1], cell[0]
            coords = coords[[1, 0, 2, 3]]
        new_coords.append(coords)
    if len({tuple(sorted(cell)) for cell in proposed}) != len(proposed):
        return None, "duplicate_new_tet"
    old_boundary = _boundary_signature(old_cells)
    new_boundary = _boundary_signature(proposed)
    if old_boundary is None or new_boundary is None or old_boundary != new_boundary:
        return None, "patch_boundary_changed"
    old_volume = sum(signed_volume(tet) for tet in old_coords)
    new_volume = sum(signed_volume(tet) for tet in new_coords)
    if abs(new_volume - old_volume) > max(512 * np.finfo(float).eps, 1e-10 * old_volume):
        return None, "patch_volume_changed"
    old_min = min(tet_quality(tet) for tet in old_coords)
    new_min = min(tet_quality(tet) for tet in new_coords)
    if not new_min > old_min + min_gain:
        return None, "no_quality_gain"
    for a, b in combinations(new_coords, 2):
        if tet_overlap(a, b, tol=_DISTANCE_TOL):
            return None, "new_tets_overlap"
    return proposed, "accepted"


def improve_topology(vertices, tets, boundary_faces, max_passes=3, min_gain=1e-8):
    """Improve patch minimum mean ratio by 2->3 and 3->2 flips.

    Return a new connectivity array and a diagnostics report.  Every accepted
    replacement preserves the exact oriented patch boundary, so domain boundary
    triangles and interfaces with all neighboring tetrahedra remain unchanged.
    Patches commit sequentially and adjacency is rebuilt after each acceptance.
    At most 128 flips per pass and 384 total flips are committed.
    """
    started = perf_counter()
    if (isinstance(max_passes, (bool, np.bool_)) or int(max_passes) != max_passes
            or not 0 <= max_passes <= 100):
        raise ValueError("max_passes must be an integer in [0, 100]")
    if not np.isfinite(min_gain) or min_gain < 0:
        raise ValueError("min_gain must be finite and nonnegative")
    points = np.asarray(vertices, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("vertices must be a finite Nx3 array")
    initial = _indices(tets, 4, len(points), "tets")
    boundary = _indices(boundary_faces, 3, len(points), "boundary_faces")
    cells = initial.copy()
    if len({tuple(sorted(cell)) for cell in cells}) != len(cells):
        raise ValueError("initial mesh contains duplicate tetrahedra")
    used = np.unique(cells)
    offset = points[used].min(axis=0)
    scale = float(np.ptp(points[used], axis=0).max())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("unsupported mesh scale")
    normalized = (points - offset) / scale
    initial_quality = np.array([tet_quality(normalized[cell]) for cell in cells])
    if np.any(initial_quality <= 0):
        raise ValueError("initial mesh has inverted or degenerate tetrahedra")
    expected_boundary = {tuple(sorted(map(int, face))): _orientation(face) for face in boundary}
    if len(expected_boundary) != len(boundary) or _boundary_signature(cells) != expected_boundary:
        raise ValueError("initial mesh boundary does not match the prescribed oriented boundary")
    before = _quality_summary(initial_quality)
    counts = {"2_to_3": 0, "3_to_2": 0}
    attempts = {"2_to_3": 0, "3_to_2": 0}
    rejections = defaultdict(int)
    total = 0
    passes_completed = 0
    moves_per_pass = []
    stop_reason = "max_passes"
    for _ in range(int(max_passes)):
        accepted_this_pass = 0
        while accepted_this_pass < _MAX_ACCEPTED_PER_PASS and total < _MAX_ACCEPTED_TOTAL:
            face_map, edge_map = _adjacency(cells)
            quality = np.array([tet_quality(normalized[cell]) for cell in cells])
            choices = []
            for face, ids in face_map.items():
                if len(ids) == 2 and face not in expected_boundary:
                    choices.append((float(quality[ids].min()), "2_to_3", face, ids))
            for edge, ids in edge_map.items():
                if len(ids) == 3:
                    choices.append((float(quality[ids].min()), "3_to_2", edge, ids))
            choices.sort(key=lambda item: (item[0], item[1], item[2]))
            accepted = False
            for _, kind, simplex, ids in choices:
                old = cells[ids]
                vertices_in_patch = set(map(int, old.ravel()))
                if len(vertices_in_patch) != 5:
                    continue
                if kind == "2_to_3":
                    a, b = sorted(vertices_in_patch - set(simplex))
                    x, y, z = simplex
                    proposed = ((a, b, x, y), (a, b, y, z), (a, b, z, x))
                else:
                    a, b = simplex
                    x, y, z = sorted(vertices_in_patch - set(simplex))
                    ring_edges = {tuple(sorted(set(map(int, cell)) - {a, b})) for cell in old}
                    if ring_edges != {(x, y), (x, z), (y, z)}:
                        continue
                    proposed = ((x, y, z, a), (x, y, z, b))
                attempts[kind] += 1
                replacement, reason = _candidate_patch(normalized, old, proposed, float(min_gain))
                if replacement is None:
                    rejections[reason] += 1
                    continue
                mask = np.ones(len(cells), dtype=bool)
                mask[ids] = False
                other_keys = {tuple(sorted(cell)) for cell in cells[mask]}
                if any(tuple(sorted(cell)) in other_keys for cell in replacement):
                    rejections["duplicates_existing_tet"] += 1
                    continue
                candidate_mesh = np.vstack((cells[mask], replacement))
                if _boundary_signature(candidate_mesh) != expected_boundary:
                    rejections["global_boundary_changed"] += 1
                    continue
                cells = candidate_mesh
                counts[kind] += 1
                accepted_this_pass += 1
                total += 1
                accepted = True
                break  # Rebuild adjacency before inspecting another patch.
            if not accepted:
                stop_reason = "no_improving_flip"
                break
        passes_completed += 1
        moves_per_pass.append(accepted_this_pass)
        if stop_reason == "no_improving_flip":
            break
        if total >= _MAX_ACCEPTED_TOTAL:
            stop_reason = "acceptance_limit"
            break
    final_quality = np.array([tet_quality(normalized[cell]) for cell in cells])
    rollback_reason = None
    if np.min(final_quality) < before["min"] or _boundary_signature(cells) != expected_boundary:
        cells = initial.copy()
        final_quality = initial_quality
        rollback_reason = "final boundary/global-quality check disagreed with local evaluations"
    report = {
        "method": "fixed-vertex 2-to-3 / 3-to-2 quality flips",
        "before": before, "after": _quality_summary(final_quality),
        "initial_tet_count": len(initial), "final_tet_count": len(cells),
        "accepted_flips": 0 if rollback_reason else total,
        "flips": {key: 0 for key in counts} if rollback_reason else counts,
        "attempts": attempts, "rejections": dict(rejections),
        "max_passes": int(max_passes), "passes_completed": passes_completed,
        "moves_per_pass": moves_per_pass,
        "acceptance_limits": {"per_pass": _MAX_ACCEPTED_PER_PASS, "total": _MAX_ACCEPTED_TOTAL},
        "min_gain": float(min_gain), "boundary_preserved": True,
        "stop_reason": stop_reason, "rollback_reason": rollback_reason,
        "seconds": float(perf_counter() - started),
        "limitations": ["local minimum-quality improvement; not all slivers can be removed",
                        "float64 tolerance predicates; validate initial and final meshes independently"],
    }
    return cells, report
