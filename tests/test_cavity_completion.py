"""Residual-only CPU recovery, including holes and disconnected volume chains."""

from collections import Counter
import importlib.util
import json
from itertools import permutations, product
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import saft3d.cavity_completion as completion
from saft3d.cavity_completion import CavityCompletionError, complete_cavity
from saft3d.cavity_worker import _plc_failure_diagnostics
from saft3d.validation import validate_mesh


def _cube(offset=(0., 0., 0.), size=1.):
    points = np.array(list(product([0., 1.], repeat=3))) * size + offset
    lookup = {tuple(p): i for i, p in enumerate(product([0, 1], repeat=3))}
    tets = []
    for axes in permutations(range(3)):
        point = np.zeros(3, dtype=int)
        tet = [lookup[tuple(point)]]
        for axis in axes:
            point[axis] = 1
            tet.append(lookup[tuple(point)])
        xyz = points[tet]
        if np.linalg.det((xyz[1:] - xyz[0]).T) < 0:
            tet[0], tet[1] = tet[1], tet[0]
        tets.append(tet)
    faces = [tuple(tet[j] for j in face) for tet in tets for face in completion._FACES]
    counts = Counter(tuple(sorted(face)) for face in faces)
    return points, np.array([face for face in faces if counts[tuple(sorted(face))] == 1])


@pytest.fixture
def tetgen_available():
    if importlib.util.find_spec("tetgen") is None:
        pytest.skip("Optional CPU residual completion requires tetgen")


def test_cube_preserves_global_prefix_and_uses_only_residual_vertices(tetgen_available):
    cube, front = _cube()
    points = np.vstack(([99., 99., 99.], cube, [-99., -99., -99.]))
    front = front + 1
    output, tets, report = complete_cavity(points, front)
    np.testing.assert_array_equal(output[:len(points)], points)
    assert report["backend"] == "cpu_tetgen" and not report["gpu"] and not report["pure_aft"]
    assert report["switches"].startswith("pYY")
    assert report["input_cavity_points"] == 8
    assert report["new_tet_count"] == len(tets)
    assert report["boundary_exact"] and report["global_prefix_unchanged"]
    assert report["cavity_volume"] == pytest.approx(1.)
    assert not np.isin(tets, [0, 9]).any()
    validation = validate_mesh(output, tets, front)
    assert validation["valid"], validation["errors"]


def test_hollow_cube_filters_inner_hole_without_seed(tetgen_available):
    outer, outer_faces = _cube()
    inner, inner_faces = _cube((.25, .25, .25), .5)
    points = np.vstack((outer, inner))
    front = np.vstack((outer_faces, (inner_faces + 8)[:, ::-1]))
    output, tets, report = complete_cavity(points, front)
    assert report["discarded_region_tets"] > 0
    assert report["cavity_volume"] == pytest.approx(1. - .5**3)
    centers = output[tets].mean(axis=1)
    assert not np.all((centers > .25) & (centers < .75), axis=1).any()
    validation = validate_mesh(output, tets, front)
    assert validation["valid"], validation["errors"]


def test_disconnected_components_do_not_fill_the_gap(tetgen_available):
    first, ff = _cube()
    second, sf = _cube((3., 0., 0.))
    points, front = np.vstack((first, second)), np.vstack((ff, sf + 8))
    output, tets, report = complete_cavity(points, front)
    assert report["cavity_volume"] == pytest.approx(2.)
    centers = output[tets].mean(axis=1)
    assert np.all((centers[:, 0] < 1.) | (centers[:, 0] > 3.))
    validation = validate_mesh(output, tets, front)
    assert validation["valid"], validation["errors"]


def test_closed_chain_with_four_faces_at_shared_edge_is_allowed(tetgen_available):
    # Two residual regions touch along an edge. Such a transient residual chain
    # need not itself be a manifold surface, although the completed full mesh is.
    first, ff = _cube()
    second, sf = _cube((1., 1., 0.))
    stacked = np.vstack((first, second))
    points, remap = np.unique(stacked, axis=0, return_inverse=True)
    front = remap[np.vstack((ff, sf + 8))]
    incidence = Counter(tuple(sorted((int(f[i]), int(f[j])))) for f in front
                        for i, j in ((0, 1), (1, 2), (2, 0)))
    assert 4 in incidence.values()
    output, tets, report = complete_cavity(points, front)
    assert report["boundary_exact"] and report["cavity_volume"] == pytest.approx(2.)
    assert completion._boundary_signature(tets) == completion._front_signature(front)
    np.testing.assert_array_equal(output[:len(points)], points)


def test_timeout_never_returns_a_successful_mesh(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(completion.subprocess, "run", timeout)
    with pytest.raises(CavityCompletionError, match="timed out"):
        complete_cavity(*_cube(), timeout=.01)


@pytest.mark.parametrize("returncode", [0, 2, -1073741819])
def test_failed_or_missing_worker_result_is_not_success(monkeypatch, returncode):
    monkeypatch.setattr(completion.subprocess, "run", lambda *args, **kwargs:
                        subprocess.CompletedProcess(args[0], returncode, "", "simulated failure"))
    with pytest.raises(CavityCompletionError):
        complete_cavity(*_cube())


def test_moved_input_prefix_is_rejected(monkeypatch):
    def corrupt_result(command, **kwargs):
        with np.load(command[3], allow_pickle=False) as data:
            nodes = data["vertices"].copy()
        nodes[0, 0] += .01
        np.savez(Path(command[4]), vertices=nodes, tetrahedra=np.array([[0, 1, 2, 3]]),
                 report=np.array("{}"))
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr(completion.subprocess, "run", corrupt_result)
    with pytest.raises(CavityCompletionError, match="prefix"):
        complete_cavity(*_cube())


def test_boundary_failure_preserves_structured_global_face_diagnostics(monkeypatch):
    diagnostics = {"failure_kind": "boundary_not_preserved", "missing_face_count": 2,
                   "missing_global_faces": [[37, 45, 47], [37, 47, 44]],
                   "boundary_steiner_local_ids": [29]}
    def rejected_boundary(command, **kwargs):
        Path(command[4]).with_suffix(".error.json").write_text(json.dumps(diagnostics), encoding="utf-8")
        return subprocess.CompletedProcess(command, 2, "", "Boundary contains Steiner points (-YY option)")
    monkeypatch.setattr(completion.subprocess, "run", rejected_boundary)
    with pytest.raises(CavityCompletionError, match="Steiner") as caught:
        complete_cavity(*_cube())
    assert caught.value.diagnostics == diagnostics


def test_boundary_edge_subdivision_reports_exact_missing_global_triangles():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    front = np.array(completion._FACES)
    global_ids = np.array([10, 20, 30, 40])
    nodes = np.vstack((points, [.5, 0., 0.]))
    cells = np.array([[0, 4, 2, 3], [4, 1, 2, 3]])
    generator = SimpleNamespace(node=nodes, elem=cells, trifaces=np.array([[0, 4, 3]]))
    diagnostics = _plc_failure_diagnostics(generator, points, front, global_ids)
    assert diagnostics["failure_kind"] == "boundary_not_preserved"
    assert diagnostics["missing_face_count"] == 2
    assert {tuple(sorted(f)) for f in diagnostics["missing_global_faces"]} == {
        (10, 20, 30), (10, 20, 40)}
    assert diagnostics["boundary_steiner_local_ids"] == [4]


def test_plc_diagnostics_use_all_tet_faces_not_only_exported_trifaces():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    generator = SimpleNamespace(node=points, elem=np.array([[0, 1, 2, 3]]),
                                trifaces=np.empty((0, 3), dtype=int))
    diagnostics = _plc_failure_diagnostics(generator, points, np.array(completion._FACES), np.arange(4))
    assert diagnostics["missing_face_count"] == 0
    assert diagnostics["missing_global_faces"] == []


def test_open_front_and_invalid_options_fail_before_worker(monkeypatch):
    def must_not_run(*args, **kwargs):
        pytest.fail("invalid input reached TetGen worker")
    monkeypatch.setattr(completion.subprocess, "run", must_not_run)
    points, front = _cube()
    with pytest.raises(CavityCompletionError, match="closed"):
        complete_cavity(points, front[:-1])
    with pytest.raises(ValueError, match="timeout"):
        complete_cavity(points, front, timeout=0)
    with pytest.raises(ValueError, match="max_steiner"):
        complete_cavity(points, front, max_steiner=-1)
