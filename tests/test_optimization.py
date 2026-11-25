"""Regression tests for fixed-connectivity, fixed-boundary quality improvement."""

import numpy as np
import pytest

from saft3d.optimization import optimize_mesh
from saft3d.surfaces import cube, tetrahedron
from saft3d.validation import validate_mesh


def _fan(surface, interior):
    vertices, boundary = surface()
    anchor = len(vertices)
    vertices = np.vstack((vertices, np.asarray(interior, dtype=float)))
    tets = np.column_stack((boundary[:, 0], boundary[:, 2], boundary[:, 1],
                           np.full(len(boundary), anchor)))
    return vertices, tets, boundary


@pytest.mark.parametrize("surface,interior", [(tetrahedron, [.3, .1, .15]),
                                              (cube, [.35, .2, -.25])])
def test_distorted_fans_improve_and_remain_valid(surface, interior):
    vertices, tets, boundary = _fan(surface, interior)
    unchanged_tets = tets.copy()
    optimized, report = optimize_mesh(vertices, tets, boundary, iterations=8)
    assert report["moves"] > 0
    assert report["after"]["min"] > report["before"]["min"] + .02
    np.testing.assert_array_equal(optimized[np.unique(boundary)], vertices[np.unique(boundary)])
    np.testing.assert_array_equal(tets, unchanged_tets)
    independent = validate_mesh(optimized, tets, boundary)
    assert independent["valid"], independent["errors"]
    assert independent["quality"]["min"] == pytest.approx(report["after"]["min"])


def test_boundary_and_unreferenced_vertices_do_not_move():
    vertices, tets, boundary = _fan(cube, [.4, .1, .1])
    vertices = np.vstack((vertices, [1e8, -1e8, 1e8]))
    before = vertices.copy()
    optimized, report = optimize_mesh(vertices, tets, boundary, iterations=3)
    np.testing.assert_array_equal(vertices, before)
    np.testing.assert_array_equal(optimized[np.unique(boundary)], before[np.unique(boundary)])
    np.testing.assert_array_equal(optimized[-1], before[-1])
    assert report["unused_vertex_count"] == 1
    assert report["moves"] > 0


def test_initial_inverted_tetrahedron_is_rejected():
    vertices, tets, boundary = _fan(tetrahedron, [0, 0, 0])
    tets[0, [0, 1]] = tets[0, [1, 0]]
    with pytest.raises(ValueError, match="inverted or degenerate"):
        optimize_mesh(vertices, tets, boundary)


def test_symmetric_optimum_returns_exact_original_coordinates():
    vertices, tets, boundary = _fan(tetrahedron, [0, 0, 0])
    optimized, report = optimize_mesh(vertices, tets, boundary)
    np.testing.assert_array_equal(optimized, vertices)
    assert report["moves"] == 0
    assert report["before"] == report["after"]


def test_zero_sweeps_is_a_validated_noop():
    vertices, tets, boundary = _fan(cube, [.2, .1, .3])
    optimized, report = optimize_mesh(vertices, tets, boundary, iterations=0)
    np.testing.assert_array_equal(optimized, vertices)
    assert report["moves"] == 0
    assert report["iterations"] == 0


@pytest.mark.parametrize("scale", [1e-7, 1e7])
def test_quality_improvement_uses_local_relative_scales(scale):
    vertices, tets, boundary = _fan(tetrahedron, [.3, .1, .15])
    vertices *= scale
    optimized, report = optimize_mesh(vertices, tets, boundary, iterations=4)
    assert report["after"]["min"] > report["before"]["min"] + .02
    np.testing.assert_array_equal(optimized[np.unique(boundary)], vertices[np.unique(boundary)])


def test_fractional_indices_are_rejected():
    vertices, tets, boundary = _fan(tetrahedron, [0, 0, 0])
    tets = tets.astype(float)
    tets[0, 0] += .2
    with pytest.raises(ValueError, match="integer indices"):
        optimize_mesh(vertices, tets, boundary)
