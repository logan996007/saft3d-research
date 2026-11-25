"""Conservative vertex relocation with fixed boundary and tetra connectivity.

Only an interior vertex may move, and only if every incident tetrahedron stays
positive and the incident minimum mean-ratio quality strictly improves.  This
is local smoothing/pattern search, not a guarantee of removing all slivers.
Input is assumed to be an otherwise valid conforming tetrahedral mesh; use the
independent validator after optimization when that has not been established.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from .backend import create_backend


_VOLUME_TOL = 64.0 * np.finfo(np.float64).eps
_QUALITY_TOL = 128.0 * np.finfo(np.float64).eps


def _indices(value, width, count, name):
    raw = np.asarray(value)
    if raw.size == 0:
        return np.empty((0, width), dtype=np.int64)
    if raw.ndim != 2 or raw.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    numeric = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError(f"{name} must contain finite integer indices")
    if np.any(numeric < 0) or np.any(numeric >= count):
        raise ValueError(f"{name} contains out-of-range indices")
    result = numeric.astype(np.int64)
    if any(len(set(row)) != width for row in result):
        raise ValueError(f"{name} contains repeated vertex indices")
    return result


def _summary(quality):
    return {"min": float(np.min(quality)),
            "p05": float(np.percentile(quality, 5)),
            "median": float(np.median(quality))}


def optimize_mesh(vertices, tets, boundary_faces, iterations=8, backend="cpu"):
    """Return relocated vertices and diagnostics without changing connectivity.

    ``iterations`` counts sweeps over internal referenced vertices.  A candidate
    batch contains neighbor-centroid and incident-volume-weighted centroid
    proposals, half steps toward those centers, and six coordinate directions.
    Directional step lengths shrink between sweeps.  CPU or CUDA ``evaluate``
    scores the whole batch; the vertex updates themselves remain sequential.

    Local stars are translated/scaled before scoring.  The normalized volume
    threshold corresponds to ``64*machine_epsilon*local_scale**3`` in the
    original coordinates.  Boundary and unused vertex values are copied exactly.
    """
    start = perf_counter()
    if isinstance(iterations, (bool, np.bool_)) or int(iterations) != iterations or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    iterations = int(iterations)
    original = np.asarray(vertices, dtype=np.float64)
    if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all():
        raise ValueError("vertices must be a finite Nx3 array")
    cells = _indices(tets, 4, len(original), "tets")
    boundary = _indices(boundary_faces, 3, len(original), "boundary_faces")
    if not len(cells):
        raise ValueError("optimization requires at least one tetrahedron")
    if not len(boundary):
        raise ValueError("optimization requires prescribed boundary faces")
    points = original.copy()
    used = np.unique(cells)
    fixed = set(map(int, boundary.ravel()))
    internal = [int(i) for i in used if int(i) not in fixed]
    origin = original[used].min(axis=0)
    global_scale = float(np.max(np.ptp(original[used], axis=0)))
    if not np.isfinite(global_scale) or global_scale <= 0.0:
        raise ValueError("mesh has zero or unsupported spatial extent")
    engine = create_backend(backend)
    normalized = (original[cells] - origin) / global_scale
    volumes, quality = engine.evaluate(normalized)
    if (not np.isfinite(volumes).all() or not np.isfinite(quality).all()
            or np.any(volumes <= 0.0) or np.any(quality <= 0.0)):
        raise ValueError("initial mesh contains inverted or degenerate tetrahedra")
    before = _summary(quality)
    incident = {vertex: [] for vertex in internal}
    for cell_id, cell in enumerate(cells):
        for vertex in cell:
            if int(vertex) in incident:
                incident[int(vertex)].append(cell_id)
    incidence = {vertex: np.asarray(ids, dtype=np.int64) for vertex, ids in incident.items()}
    neighbors = {vertex: np.asarray(sorted(set(map(int, cells[ids].ravel())) - {vertex}), dtype=np.int64)
                 for vertex, ids in incidence.items()}
    moves = 0
    moved_vertices = set()
    evaluated_candidates = 0
    sweep_moves = []
    # Volumes are retained in the initial global normalization so their relative
    # weights are consistent when adjacent stars have different local scales.
    current_volumes = volumes.copy()
    for sweep in range(iterations):
        accepted_this_sweep = 0
        directional_step = 0.2 * 2.0 ** (-sweep / 2.0)
        for vertex in internal:
            ids = incidence[vertex]
            center = points[vertex].copy()
            displacement = points[neighbors[vertex]] - center
            scale = float(np.linalg.norm(displacement, axis=1).max())
            if not np.isfinite(scale) or scale <= 0.0:
                continue
            local = (points[cells[ids]] - center) / scale
            neighbor_center = (displacement / scale).mean(axis=0)
            weights = current_volumes[ids]
            weighted_center = np.average(local.mean(axis=1), axis=0, weights=weights)
            proposals = [np.zeros(3), neighbor_center, .5 * neighbor_center,
                         weighted_center, .5 * weighted_center]
            proposals.extend(directional_step * axis for axis in np.vstack((np.eye(3), -np.eye(3))))
            candidate_points = []
            candidate_local = []
            for proposal in proposals:
                # Evaluate the representable output point, not an idealized
                # normalized point that could round differently when stored.
                point = center + scale * proposal
                if not np.isfinite(point).all():
                    continue
                displacement_local = (point - center) / scale
                if any(np.array_equal(point, prior) for prior in candidate_points):
                    continue
                candidate_points.append(point)
                candidate_local.append(displacement_local)
            if len(candidate_points) < 2:
                continue
            candidate_local = np.asarray(candidate_local)
            batch = np.broadcast_to(local, (len(candidate_local),) + local.shape).copy()
            slots = np.argmax(cells[ids] == vertex, axis=1)
            batch[:, np.arange(len(ids)), slots, :] = candidate_local[:, None, :]
            candidate_volume, candidate_quality = engine.evaluate(batch.reshape(-1, 4, 3))
            candidate_volume = candidate_volume.reshape(len(candidate_local), len(ids))
            candidate_quality = candidate_quality.reshape(len(candidate_local), len(ids))
            evaluated_candidates += len(candidate_local) - 1
            legal = (np.all(candidate_volume > _VOLUME_TOL, axis=1)
                     & np.all(np.isfinite(candidate_volume), axis=1)
                     & np.all(np.isfinite(candidate_quality), axis=1))
            local_min = candidate_quality.min(axis=1)
            baseline = float(local_min[0])
            legal[0] = False
            legal &= local_min > baseline + _QUALITY_TOL * max(1.0, abs(baseline))
            if not np.any(legal):
                continue
            best = int(np.argmax(np.where(legal, local_min, -np.inf)))
            points[vertex] = candidate_points[best]
            current_volumes[ids] = candidate_volume[best] * (scale / global_scale) ** 3
            moves += 1
            accepted_this_sweep += 1
            moved_vertices.add(vertex)
        sweep_moves.append(accepted_this_sweep)
    final_volumes, final_quality = engine.evaluate((points[cells] - origin) / global_scale)
    rollback_reason = None
    if (not np.isfinite(final_volumes).all() or not np.isfinite(final_quality).all()
            or np.any(final_volumes <= 0.0) or np.min(final_quality) < before["min"]):
        # Fail closed if differently scaled floating-point evaluations disagree.
        points = original.copy()
        final_quality = quality
        rollback_reason = "final global positivity/quality check disagreed with local evaluations"
    committed_moves = 0 if rollback_reason else moves
    report = {
        "method": "fixed-boundary interior-vertex mean-ratio pattern search",
        "before": before, "after": _summary(final_quality),
        "moves": committed_moves,
        "moved_vertex_count": 0 if rollback_reason else len(moved_vertices),
        "iterations": iterations, "moves_per_sweep": sweep_moves,
        "interior_vertex_count": len(internal),
        "fixed_boundary_vertex_count": len(fixed),
        "unused_vertex_count": len(original) - len(used),
        "evaluated_candidates": evaluated_candidates,
        "backend": engine.name, "backend_stats": engine.stats,
        "kernel_counts": dict(engine.stats.get("kernel_calls", {})),
        "rollback_reason": rollback_reason,
        "seconds": float(perf_counter() - start),
        "guarantee_scope": "accepted moves preserve positive incident volumes and improve their minimum mean ratio; no guarantee of eliminating all slivers",
    }
    return points, report
