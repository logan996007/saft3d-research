from itertools import permutations

import numpy as np
import pytest

from fem_comparison.elasticity import (
    POISSON, TRACTION, YOUNG, _solve_static, _strain_displacement, _surface_weights, solve,
)


def _unit_bricks(corner_ids):
    """Freudenthal tetrahedra and their external triangles, built from unit bricks."""
    point_ids, cells = {}, []
    for corner in corner_ids:
        for perm in permutations(range(3)):
            p = np.asarray(corner, dtype=int).copy()
            cell = [point_ids.setdefault(tuple(p), len(point_ids))]
            for k in perm:
                p[k] += 1
                cell.append(point_ids.setdefault(tuple(p), len(point_ids)))
            cells.append(cell)
    points = np.asarray(list(point_ids), dtype=float)
    cells = np.array(cells)
    inverted = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]]) < 0
    cells[inverted] = cells[inverted][:, [0, 2, 1, 3]]
    triangles = {}
    for a, b, c, d in cells:
        for face in [(a, c, b), (a, b, d), (a, d, c), (b, c, d)]:
            key = tuple(sorted(face))
            if key in triangles: del triangles[key]
            else: triangles[key] = face
    return points, cells, np.array(list(triangles.values()))


def test_affine_uniaxial_tension_patch_with_free_poisson_contraction():
    points, cells, faces = _unit_bricks([(0, 0, 0)])
    # The roller planes permit the exact transverse contraction.  The extra
    # point represents a discarded meshing proposal and must carry no DOFs.
    used = len(points)
    points = np.vstack([points, [3., 3., 3.]])
    right = faces[(points[faces, 0] == 1).all(axis=1)]
    loads = np.zeros_like(points)
    loads[:, 0] = TRACTION * _surface_weights(points, right)
    metrics, fields = _solve_static(points, cells, fixed=(points == 0), nodal_forces=loads)
    expected = (TRACTION / YOUNG) * points[:used] * np.array([1., -POISSON, -POISSON])
    np.testing.assert_allclose(fields['displacement'][:used], expected, atol=1e-16)
    np.testing.assert_array_equal(fields['displacement'][-1], [0, 0, 0])
    np.testing.assert_allclose(fields['von_mises'], TRACTION, rtol=1e-12)
    assert metrics['active_vertices'] == used and metrics['free_dofs'] == 12
    assert metrics['compliance_J'] == pytest.approx(TRACTION**2 / YOUNG, rel=1e-12)
    assert metrics['strain_energy_J'] == pytest.approx(0.5 * TRACTION**2 / YOUNG, rel=1e-12)


def test_rigid_rotation_has_zero_shear_and_volume_strain():
    points, cells, _ = _unit_bricks([(0, 0, 0)])
    volumes, bmat = _strain_displacement(points, cells)
    u = np.cross([.23, .78, -.47], points) + [4., 7., -3.]
    strain = np.einsum('tij,tj->ti', bmat, u[cells].reshape(-1, 12))
    np.testing.assert_allclose(strain, 0, atol=3e-15)
    assert volumes.sum() == pytest.approx(1.)


def test_u_bracket_load_clamp_and_work_balance():
    points, cells, faces = _unit_bricks([(x,y,0) for y in range(3) for x in range(3)
                                         if y == 0 or x != 1])
    points[:, 2] -= .5
    metrics, fields = solve(points,cells,faces)
    assert metrics['loaded_area_m2'] == pytest.approx(1.)
    assert metrics['clamped_area_m2'] == pytest.approx(1.)
    assert metrics['mean_loaded_ux_m'] > 0
    assert metrics['compliance_J'] == pytest.approx(TRACTION * metrics['mean_loaded_ux_m'])
    assert metrics['energy_identity_relative'] < 1e-10
    assert metrics['free_residual_relative'] < 1e-10
    np.testing.assert_allclose(metrics['reaction_N'], [-1e6, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(fields['displacement'][(points[:,1]==3)&(points[:,0]<=1)], 0)
