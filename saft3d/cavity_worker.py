"""Subprocess-only CPU TetGen adapter; receives and returns non-pickled NPZ data."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

from .cavity_completion import (
    CavityCompletionError, _FACES, _boundary_signature, _front_signature,
    _signed_surface_volume, _volumes,
)
from .front_geometry import FrontGeometry


def _plc_failure_diagnostics(generator, points, front, global_ids):
    """Inspect a rejected TetGen complex, never authorize using that complex.

    TetGen 0.8.4 saves partial output before raising for remaining boundary
    Steiner points under -YY. Missing global triangles identify nearby accepted
    AFT cells that a caller may explicitly roll back before retrying a larger
    residual cavity. No boundary subdivision is silently accepted here.
    """
    diagnostics = {"failure_kind": "tetgen_failure"}
    try:
        nodes, cells = generator.node, generator.elem
        all_faces = {tuple(sorted(int(tet[j]) for j in local))
                     for tet in cells for local in _FACES}
        missing = [i for i, face in enumerate(front) if tuple(sorted(map(int, face))) not in all_faces]
        diagnostics.update({
            "missing_face_count": len(missing),
            "missing_front_indices": missing[:50],
            "missing_faces_local": front[missing[:50]].tolist(),
            "missing_global_faces": global_ids[front[missing[:50]]].tolist(),
            "input_prefix_unchanged": bool(len(nodes) >= len(points)
                and np.array_equal(nodes[:len(points)], points)),
        })
        if missing:
            diagnostics["failure_kind"] = "boundary_not_preserved"
        # This is diagnostic only. The acceptance check above/below inspects
        # every tetrahedron face, not just TetGen's exported boundary faces.
        surface_ids = np.unique(generator.trifaces)
        diagnostics["boundary_steiner_local_ids"] = surface_ids[surface_ids >= len(points)].tolist()[:50]
    except Exception as exc:
        diagnostics["partial_output_unavailable"] = str(exc)
    return diagnostics


def run_worker(input_path, output_path, max_steiner):
    """Run CPU recovery and classify PLC regions by the original front winding."""
    import tetgen

    start = perf_counter()
    with np.load(input_path, allow_pickle=False) as data:
        points = np.ascontiguousarray(data["vertices"], dtype=np.float64)
        front = np.ascontiguousarray(data["front"], dtype=np.int32)
        global_ids = np.asarray(data["global_ids"], dtype=np.int64) if "global_ids" in data else np.arange(len(points))
    geometry = FrontGeometry(points, front)
    markers = np.arange(1, len(front) + 1, dtype=np.int32)
    # Single Y only tries to suppress boundary Steiner points. Double YY also
    # makes TetGen reject the result if suppression fails; our exact interface
    # check remains mandatory even when TetGen reports success.
    switches = f"pYYJMQzO0S{max_steiner}"
    generator = tetgen.TetGen(points, front, markers)
    try:
        nodes, cells, _, _ = generator.tetrahedralize(switches=switches)
    except Exception as exc:
        diagnostics = _plc_failure_diagnostics(generator, points, front, global_ids)
        diagnostics.update({"switches": switches, "tetgen_error": str(exc)})
        raise CavityCompletionError(str(exc), diagnostics=diagnostics) from exc
    nodes = np.asarray(nodes, dtype=np.float64)
    cells = np.asarray(cells, dtype=np.int64)
    if (nodes.ndim != 2 or nodes.shape[1:] != (3,) or len(nodes) < len(points)
            or not np.isfinite(nodes).all() or not np.array_equal(nodes[:len(points)], points)):
        raise CavityCompletionError("TetGen did not preserve the local input node prefix")
    if len(nodes) - len(points) > max_steiner:
        raise CavityCompletionError("TetGen exceeded the requested added-node budget")
    if (cells.ndim != 2 or cells.shape[1:] != (4,) or not len(cells)
            or np.any(cells < 0) or np.any(cells >= len(nodes))):
        raise CavityCompletionError("TetGen produced no valid linear tetrahedra")

    # Check the full constrained complex before filtering its material regions.
    # Every PLC triangle must remain an actual tet face, with the original IDs.
    all_faces = {tuple(sorted(int(tet[j]) for j in local)) for tet in cells for local in _FACES}
    prescribed = _front_signature(front)
    if not set(prescribed).issubset(all_faces):
        diagnostics = _plc_failure_diagnostics(generator, points, front, global_ids)
        diagnostics["switches"] = switches
        raise CavityCompletionError("TetGen did not preserve every input PLC triangle exactly",
                                    diagnostics=diagnostics)
    raw_count = len(cells)
    centroids = nodes[cells].mean(axis=1)
    keep = np.array([geometry.contains(center) for center in centroids], dtype=bool)
    cells = cells[keep].copy()
    if not len(cells):
        raise CavityCompletionError("No tetrahedra lie in the residual material winding region")
    volumes = _volumes(nodes, cells)
    if not np.isfinite(volumes).all() or np.any(volumes == 0):
        raise CavityCompletionError("TetGen generated degenerate tetrahedra")
    inverted = volumes < 0
    cells[inverted, 0], cells[inverted, 1] = cells[inverted, 1].copy(), cells[inverted, 0].copy()
    if _boundary_signature(cells) != prescribed:
        raise CavityCompletionError("Filtered residual boundary differs in triangle IDs or orientation")
    expected = _signed_surface_volume(points, front)
    if expected <= 0 or abs(float(np.abs(volumes).sum()) - expected) / expected > 1.e-7:
        raise CavityCompletionError("Filtered residual mesh has incorrect signed volume")

    # Retain the complete local input prefix, but append only used Steiner nodes.
    added_ids = np.unique(cells[cells >= len(points)])
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[:len(points)] = np.arange(len(points))
    remap[added_ids] = np.arange(len(points), len(points) + len(added_ids))
    cells = remap[cells]
    output_nodes = np.concatenate((points, nodes[added_ids]))
    report = {
        "tetgen_version": tetgen.__version__,
        "switches": switches,
        "tetgen_raw_tet_count": raw_count,
        "discarded_region_tets": int((~keep).sum()),
        "tetgen_added_point_count": len(nodes) - len(points),
        "region_filter": "FrontGeometry.contains: residual oriented winding near 1; holes near 0 discarded",
        "plc_triangles_preserved": True,
        "worker_seconds": perf_counter() - start,
    }
    np.savez_compressed(output_path, vertices=output_nodes, tetrahedra=cells,
                        report=np.array(json.dumps(report, allow_nan=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-steiner", type=int, default=10000)
    args = parser.parse_args()
    try:
        run_worker(args.input, args.output, args.max_steiner)
    except Exception as exc:
        print(f"CPU cavity completion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        diagnostics = getattr(exc, "diagnostics", {})
        if diagnostics:
            print("Cavity diagnostics: " + json.dumps(diagnostics), file=sys.stderr)
            try:
                args.output.with_suffix(".error.json").write_text(
                    json.dumps(diagnostics, allow_nan=False), encoding="utf-8")
            except (OSError, ValueError):
                pass
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
