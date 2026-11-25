"""Bounded end-to-end regressions for local 3D advancing-front generation.

These exercise success, explicit partial failure and real CUDA execution without
running the expensive full hollow-domain experiment or requiring TetGen success.
"""

import importlib.util

import numpy as np
import pytest

from saft3d.backend import create_backend
from saft3d.front_geometry import FrontGeometry
from saft3d.general_mesher import AdvancingFrontConfig, StalledFront, grow_general_mesh
from saft3d.general_surfaces import make_general_surface
from saft3d.geometry import signed_volume
from saft3d.surfaces import cube, tetrahedron
from saft3d.validation import validate_mesh


@pytest.fixture(scope="module")
def cuda_backend():
    if importlib.util.find_spec("cupy") is None:
        pytest.skip("CUDA unavailable: CuPy is not installed")
    import cupy as cp

    try:
        count = cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        if exc.status in (3, 35, 100):
            pytest.skip(f"CUDA unavailable: {exc}")
        raise
    if count == 0:
        pytest.skip("CUDA unavailable: no device detected")
    # Compilation/kernel failures on an available device are actual failures.
    return create_backend("cuda")


def _no_new_config(**overrides):
    return AdvancingFrontConfig(**{
        "backend": "cpu", "max_new_vertices": 0, "completion": "none",
        "max_iterations": 100, **overrides,
    })


@pytest.mark.parametrize("shape,new_vertex_limit", [(tetrahedron, 0), (cube, 1)],
                         ids=["tetrahedron", "cube"])
def test_cpu_local_aft_finishes_small_domains_and_validates_independently(shape, new_vertex_limit):
    vertices, faces = shape()
    points, tets, run = grow_general_mesh(
        vertices, faces, _no_new_config(max_new_vertices=new_vertex_limit))
    assert run["remaining_front_faces"] == 0
    assert run["growth_tets"] == len(tets) > 0
    assert run["closure_tets"] == 0
    assert run["new_vertices"] == len(points) - len(vertices) <= new_vertex_limit
    assert "cpu_completion" not in run
    assert "kernel_point" not in run
    assert all(signed_volume(tet) > 0 for tet in points[tets])
    np.testing.assert_array_equal(points[:len(vertices)], vertices)
    report = validate_mesh(points, tets, faces)
    assert report["valid"], report["errors"]
    assert report["positive_tet_count"] == len(tets)
    assert report["boundary_faces_match"]
    assert report["relative_volume_error"] < 1e-10


def test_u_prism_pure_aft_stall_retains_honest_nonoverlapping_partial():
    vertices, faces, metadata = make_general_surface("u_prism")
    with pytest.raises(StalledFront, match="stalled.*front faces") as caught:
        grow_general_mesh(vertices, faces, _no_new_config())
    partial = caught.value
    assert len(partial.tetrahedra) > 0
    assert len(partial.front) == partial.report["remaining_front_faces"] > 0
    assert partial.report["growth_tets"] == len(partial.tetrahedra)
    assert partial.report["closure_tets"] == 0
    assert partial.report["config"]["completion"] == "none"
    assert "cpu_completion" not in partial.report
    assert partial.report.get("success") is not True
    assert all(signed_volume(tet) > 0 for tet in partial.vertices[partial.tetrahedra])
    # This partial must NOT validate as a completed mesh of the original domain.
    report = validate_mesh(partial.vertices, partial.tetrahedra, faces)
    assert not report["valid"]
    assert not report["boundary_faces_match"]
    assert report["positive_tet_count"] == len(partial.tetrahedra)
    assert 0 < report["positive_volume_sum"] < metadata["analytic_volume"]
    assert report["overlap_count"] == 0
    assert report["nonconforming_contact_count"] == 0
    assert report["intersection_failure_count"] == 0
    assert report["internal_orientation_error_count"] == 0


def test_multishell_cavity_is_excluded_during_one_local_aft_iteration():
    vertices, faces, metadata = make_general_surface("hollow_cube")
    with pytest.raises(StalledFront) as caught:
        grow_general_mesh(vertices, faces, _no_new_config(max_iterations=1))
    partial = caught.value
    assert partial.report["initial_volume"] == pytest.approx(metadata["analytic_volume"])
    assert partial.report["remaining_front_faces"] > 0
    assert len(partial.tetrahedra) > 0
    front = FrontGeometry(vertices, faces)
    assert front.contains([.8, .8, .8])
    assert not front.contains(metadata["hole_seeds"][0])
    for tet in partial.vertices[partial.tetrahedra]:
        assert front.contains(tet.mean(axis=0))
        assert signed_volume(tet) > 0
    remaining = FrontGeometry(partial.vertices, partial.front)
    assert not remaining.contains(metadata["hole_seeds"][0])
    assert not remaining.contains(partial.vertices[partial.tetrahedra[0]].mean(axis=0))


def test_invalid_surfaces_are_rejected_before_generation():
    vertices, faces = cube()
    with pytest.raises(ValueError, match="closed"):
        grow_general_mesh(vertices, faces[:-1], _no_new_config())
    with pytest.raises(ValueError, match="integer indices"):
        grow_general_mesh(vertices, faces.astype(float), _no_new_config())
    with pytest.raises(ValueError, match="positive outward signed volume"):
        grow_general_mesh(vertices, faces[:, ::-1], _no_new_config())
    duplicated = np.vstack((vertices, vertices[0]))
    with pytest.raises(ValueError, match="duplicate or near-coincident"):
        grow_general_mesh(duplicated, faces, _no_new_config())


def test_general_configuration_errors_are_explicit():
    vertices, faces = tetrahedron()
    cases = [
        ({"size": 0}, "size"),
        ({"candidate_vertices": 0}, "candidate_vertices"),
        ({"batch_size": 513}, "batch_size"),
        ({"min_quality": .5, "new_vertex_min_quality": .1}, "quality thresholds"),
        ({"max_new_vertices": -1}, "resource limits"),
        ({"completion": "silent_fallback"}, "completion"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            grow_general_mesh(vertices, faces, _no_new_config(**changes))


def test_real_cuda_and_cpu_match_on_small_tetrahedral_domain(cuda_backend, monkeypatch):
    vertices, faces = tetrahedron()
    cpu_points, cpu_tets, _ = grow_general_mesh(vertices, faces, _no_new_config())
    # Reuse the fixture's real device backend; no numerical kernel is mocked.
    import saft3d.general_mesher as general_module
    real_factory = general_module.create_backend
    monkeypatch.setattr(general_module, "create_backend",
                        lambda name: cuda_backend if name == "cuda" else real_factory(name))
    gpu_points, gpu_tets, run = grow_general_mesh(vertices, faces, _no_new_config(backend="cuda"))
    np.testing.assert_allclose(gpu_points, cpu_points, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(gpu_tets, cpu_tets)
    assert run["backend"] == "cuda"
    assert run["remaining_front_faces"] == 0
    assert run["backend_stats"]["fallback_reason"] is None
    for kernel in ("propose_local_tets", "evaluate_tets", "aabb_conflicts"):
        assert run["backend_stats"]["kernel_calls"][kernel] > 0
    report = validate_mesh(gpu_points, gpu_tets, faces)
    assert report["valid"], report["errors"]


def test_cpu_completion_failure_preserves_partial_diagnostics(monkeypatch):
    vertices, faces = cube()

    def completion_failed(*args, **kwargs):
        raise RuntimeError("simulated frozen-interface completion failure")

    monkeypatch.setattr("saft3d.cavity_completion.complete_cavity", completion_failed)
    with pytest.raises(StalledFront) as caught:
        grow_general_mesh(vertices, faces,
                          _no_new_config(max_iterations=1, completion="tetgen"))
    partial = caught.value
    assert partial.report["remaining_front_faces"] == len(partial.front) > 0
    assert len(partial.tetrahedra) == partial.report["growth_tets"] > 0
    assert partial.report["closure_tets"] == 0
    assert "simulated frozen-interface completion failure" in partial.report["completion_error"]
    assert "cpu_completion" not in partial.report
    assert partial.report.get("success") is not True
