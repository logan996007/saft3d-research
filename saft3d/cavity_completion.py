"""Isolated CPU TetGen completion of an already constructed residual front.

This is an explicit constrained-Delaunay fallback, not GPU advancing front.
Only the residual front is submitted to TetGen; existing volume cells are not
remeshed. Results are returned transactionally after exact indexed-boundary and
original-coordinate-prefix checks. A caller must still independently validate
the union with its previously accepted tetrahedra.
"""

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np


_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


class CavityCompletionError(RuntimeError):
    """Failed/unfinished completion; no replacement mesh has been committed."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


def _face_sign(face):
    return -1 if sum(face[i] > face[j] for i in range(3) for j in range(i + 1, 3)) % 2 else 1


def _boundary_signature(tetrahedra):
    """Extract an oriented boundary, rejecting invalid internal incidences."""
    entries = defaultdict(list)
    for tet in tetrahedra:
        for local in _FACES:
            face = tuple(int(tet[j]) for j in local)
            entries[tuple(sorted(face))].append(_face_sign(face))
    boundary = {}
    for key, signs in entries.items():
        if len(signs) == 1:
            boundary[key] = signs[0]
        elif len(signs) != 2 or sum(signs) != 0:
            raise CavityCompletionError("Completion has invalid internal face incidence/orientation")
    return boundary


def _front_signature(front):
    result = {}
    for row in front:
        face = tuple(map(int, row))
        key = tuple(sorted(face))
        if key in result:
            raise CavityCompletionError("Residual front contains duplicate triangles")
        result[key] = _face_sign(face)
    return result


def _volumes(points, tets):
    coords = points[tets]
    u, v, w = (coords[:, i] - coords[:, 0] for i in (1, 2, 3))
    return np.einsum("ij,ij->i", u, np.cross(v, w)) / 6.


def _signed_surface_volume(points, front):
    shifted = points - points.mean(axis=0)
    tri = shifted[front]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.)


def _validate_inputs(vertices, front):
    points = np.asarray(vertices, dtype=np.float64)
    raw = np.asarray(front)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError("vertices must be a finite Nx3 array")
    if raw.ndim != 2 or raw.shape[1:] != (3,) or not len(raw):
        raise ValueError("front must be a nonempty Fx3 array")
    numeric = np.asarray(raw, dtype=float)
    if (not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all()
            or np.any(numeric < 0) or np.any(numeric >= len(points))):
        raise ValueError("front contains invalid vertex indices")
    faces = numeric.astype(np.int64)
    if any(len(set(map(int, face))) != 3 for face in faces):
        raise ValueError("front contains a triangle with repeated indices")
    _front_signature(faces)
    edges = Counter()
    for face in faces:
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(face[i]), int(face[j])
            edges[tuple(sorted((a, b)))] += 1 if a < b else -1
    # A residual volume chain can have 4 (or more even) face incidences around
    # a pinched edge. Requiring exactly two would reject legal residual cavities.
    if any(edges.values()):
        raise CavityCompletionError("Residual front is not a closed oriented chain")
    used, local_faces = np.unique(faces, return_inverse=True)
    local_faces = local_faces.reshape(-1, 3)
    local_points = points[used]
    if len(np.unique(local_points, axis=0)) != len(local_points):
        raise CavityCompletionError("Residual front has duplicate coordinates with distinct IDs")
    scale = float(np.max(np.ptp(local_points, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise CavityCompletionError("Residual cavity has zero or invalid spatial extent")
    offset = local_points.min(axis=0)
    normalized = (local_points - offset) / scale
    cavity_volume = _signed_surface_volume(normalized, local_faces)
    if not np.isfinite(cavity_volume) or cavity_volume <= 0:
        raise CavityCompletionError("Residual front has nonpositive signed volume")
    return points, faces, used, normalized, local_faces, offset, scale, cavity_volume


def complete_cavity(vertices, front, timeout=60, max_steiner=10000):
    """Fill only residual material, returning global points, new tets, and report.

    The global vertex prefix and its IDs remain unchanged. Existing volume cells
    must be retained by the caller. ``front`` is an outward oriented chain of
    triangles; nested inner shells face into holes. Internal voids are selected
    by winding after constrained meshing, without guessing hole seed points.
    Failure, a changed boundary, or timeout raises ``CavityCompletionError``.
    """
    start = perf_counter()
    timeout = float(timeout)
    if not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    if isinstance(max_steiner, bool) or int(max_steiner) != max_steiner or max_steiner < 0:
        raise ValueError("max_steiner must be a nonnegative integer")
    max_steiner = int(max_steiner)
    points, faces, used, normalized, local_faces, offset, scale, expected_volume = _validate_inputs(vertices, front)
    root = Path(__file__).resolve().parent.parent
    logs = ""
    with TemporaryDirectory(prefix="saft3d-cavity-") as temporary:
        input_path, output_path = Path(temporary) / "input.npz", Path(temporary) / "output.npz"
        np.savez(input_path, vertices=normalized, front=local_faces, global_ids=used)
        command = [sys.executable, "-m", "saft3d.cavity_worker", str(input_path),
                   str(output_path), "--max-steiner", str(max_steiner)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            completed = subprocess.run(command, cwd=root, capture_output=True, text=True,
                                       encoding="utf-8", errors="replace", timeout=timeout,
                                       creationflags=flags, check=False)
        except subprocess.TimeoutExpired as exc:
            raise CavityCompletionError(f"CPU TetGen residual-cavity completion timed out after {timeout:g}s") from exc
        except OSError as exc:
            raise CavityCompletionError(f"Cannot launch CPU cavity worker: {exc}") from exc
        logs = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()[-4000:]
        if completed.returncode != 0:
            diagnostics = {}
            error_path = output_path.with_suffix(".error.json")
            if error_path.is_file():
                try:
                    payload = json.loads(error_path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        diagnostics = payload
                except (OSError, ValueError):
                    pass
            raise CavityCompletionError(f"CPU cavity worker failed (exit {completed.returncode}): {logs}",
                                        diagnostics=diagnostics)
        if not output_path.is_file():
            raise CavityCompletionError("CPU cavity worker exited without a result")
        try:
            with np.load(output_path, allow_pickle=False) as data:
                local_output = np.array(data["vertices"], dtype=float, copy=True)
                local_tets = np.array(data["tetrahedra"], copy=True)
                metadata = json.loads(str(data["report"].item()))
        except Exception as exc:
            raise CavityCompletionError(f"Invalid CPU cavity worker result: {exc}") from exc

    # Do not trust a successful subprocess exit as proof of a usable mesh.
    if (local_output.ndim != 2 or local_output.shape[1:] != (3,)
            or not np.isfinite(local_output).all() or len(local_output) < len(used)
            or not np.array_equal(local_output[:len(used)], normalized)):
        raise CavityCompletionError("TetGen changed the input coordinate prefix")
    if (local_tets.ndim != 2 or local_tets.shape[1:] != (4,) or not len(local_tets)
            or local_tets.dtype.kind not in "iu" or np.any(local_tets < 0)
            or np.any(local_tets >= len(local_output))):
        raise CavityCompletionError("CPU cavity worker returned invalid or empty tetrahedra")
    if len(local_output) - len(used) > max_steiner:
        raise CavityCompletionError("CPU cavity result exceeds the Steiner-point budget")
    if _boundary_signature(local_tets) != _front_signature(local_faces):
        raise CavityCompletionError("CPU cavity result changed residual boundary keys or orientation")
    volumes = _volumes(local_output, local_tets)
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise CavityCompletionError("CPU cavity result contains nonpositive tetrahedra")
    relative_error = abs(float(volumes.sum()) - expected_volume) / expected_volume
    if relative_error > 1.e-7:
        raise CavityCompletionError("CPU cavity result does not conserve residual volume")

    added = local_output[len(used):] * scale + offset
    output_points = np.concatenate((points.copy(), added))
    mapping = np.concatenate((used, np.arange(len(points), len(output_points), dtype=np.int64)))
    new_tets = mapping[local_tets]
    if not np.array_equal(output_points[:len(points)], points):
        raise CavityCompletionError("Original global coordinates were modified")
    if _boundary_signature(new_tets) != _front_signature(faces):
        raise CavityCompletionError("Global vertex mapping changed the residual boundary")
    physical_volumes = _volumes(output_points, new_tets)
    if not np.isfinite(physical_volumes).all() or np.any(physical_volumes <= 0):
        raise CavityCompletionError("Global-coordinate conversion invalidated a tetrahedron")
    report = {
        **metadata,
        "backend": "cpu_tetgen",
        "method": "CPU constrained-Delaunay residual-cavity completion",
        "gpu": False,
        "pure_aft": False,
        "original_global_points": len(points),
        "input_cavity_points": len(used),
        "input_front_faces": len(faces),
        "new_tet_count": len(new_tets),
        "new_point_count": len(added),
        "cavity_volume": expected_volume * scale**3,
        "new_tet_volume": float(physical_volumes.sum()),
        "relative_volume_error": relative_error,
        "boundary_exact": True,
        "global_prefix_unchanged": True,
        "wall_seconds": perf_counter() - start,
        "logs": logs,
        "validation_scope": "Residual PLC and boundary checked; caller must validate union with existing tets",
    }
    return output_points, new_tets, report
