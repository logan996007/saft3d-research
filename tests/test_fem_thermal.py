from collections import Counter
from itertools import combinations, permutations, product

import numpy as np
import pytest

from fem_comparison.thermal import _operators, solve


def uniform_cube(n):
    grid = np.asarray(list(product(range(n+1), repeat=3)))
    points = grid / n
    ids = {tuple(point): i for i, point in enumerate(grid)}
    eye = np.eye(3, dtype=int)
    cells = []
    for origin in product(range(n), repeat=3):
        for order in permutations(range(3)):
            path = np.vstack([np.zeros(3, dtype=int), np.cumsum(eye[list(order)], axis=0)]) + origin
            tet = [ids[tuple(point)] for point in path]
            if np.linalg.det(points[tet][1:] - points[tet[0]]) < 0:
                tet[1], tet[2] = tet[2], tet[1]
            cells.append(tet)
    counts = Counter(tuple(sorted(f)) for c in cells for f in combinations(c, 3))
    return points, np.array(cells), np.array([f for f, nshare in counts.items() if nshare == 1])


def test_gradient_and_energy_for_affine_patch():
    v, t, _ = uniform_cube(2)
    vols, grads, matrix, load = _operators(v, t)
    slope = np.array([2.3, -.6, 1.2])
    field = 2.0 + v @ slope
    evaluated = np.einsum("ei,eij->ej", field[t], grads)
    np.testing.assert_allclose(evaluated, np.tile(slope, (len(t), 1)), atol=1e-12)
    assert field @ matrix @ field == pytest.approx(10.0 * sum(slope**2))
    assert sum(load) == pytest.approx(10_000.0 * sum(vols))


def test_coarsest_mesh_has_exact_unit_relative_errors():
    v, t, f = uniform_cube(1)
    result, fields = solve(np.vstack((v, [0, 0, 1000])), t, f)
    assert result['free_dofs'] == 0
    assert result['used_nodes'] == 8
    assert result['relative_l2_temperature_rise'] == pytest.approx(1.0)
    assert result['relative_l2_heat_flux'] == pytest.approx(1.0)
    assert np.all(fields['temperature'] == 300.0)


def test_refinement_reduces_continuous_errors_and_conserves_heat():
    results = [solve(*uniform_cube(n))[0] for n in [2, 4, 8]]
    for r in results:
        assert r['relative_heat_imbalance'] < 1e-12
        assert r['top_heat_out_W'] == pytest.approx(5000.0)
        assert r['bottom_heat_out_W'] == pytest.approx(5000.0)
        # Galerkin orthogonality for homogeneous cap data:
        # relative squared energy error = 1 - int(T_h-300)/int(T_exact-300).
        assert r['relative_l2_heat_flux']**2 == pytest.approx(
            1 - r['mean_rise_K'] / r['exact_mean_rise_K'], abs=1e-12)
    for coarse, fine in zip(results, results[1:]):
        assert fine['relative_l2_temperature_rise'] < .40 * coarse['relative_l2_temperature_rise']
        assert fine['relative_l2_heat_flux'] < .60 * coarse['relative_l2_heat_flux']
