"""Separating-plane certificates must preserve independent LP decisions."""

from collections import Counter
from itertools import combinations, permutations, product

import numpy as np
import pytest

import saft3d.validation as validation


_UNIT = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
_IDS = np.arange(4)


def _boundary(tets):
    faces = [tuple(int(tet[j]) for j in face) for tet in tets for face in validation._OUTWARD_FACES]
    counts = Counter(tuple(sorted(face)) for face in faces)
    return np.array([face for face in faces if counts[tuple(sorted(face))] == 1])


@pytest.mark.parametrize("shared_count", [1, 2, 3])
def test_random_affine_conforming_pairs_match_original_lp(shared_count):
    rng = np.random.default_rng(7038 + shared_count)
    if shared_count == 1:
        other, ids = -_UNIT, np.array([0, 4, 5, 6])
    elif shared_count == 2:
        other, ids = _UNIT * [1., -1., -1.], np.array([0, 1, 4, 5])
    else:
        other, ids = _UNIT * [1., 1., -1.], np.array([0, 1, 2, 4])
    for _ in range(8):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        affine = rotation @ np.diag(rng.uniform(.4, 2., 3))
        translation = rng.uniform(-2., 2., 3)
        a, b = _UNIT @ affine.T + translation, other @ affine.T + translation
        original, old_calls = validation._intersection_kind(a, b, _IDS, ids)
        kind, calls, certified = validation._intersection_with_certificate(a, b, _IDS, ids)
        assert original == kind == "conforming"
        assert old_calls == 2 and calls == 0 and certified


def test_random_unshared_pairs_compare_with_original_lp():
    rng = np.random.default_rng(209)
    for _ in range(24):
        a = rng.uniform(-.3, .3, (4, 3))
        b = rng.uniform(-.3, .3, (4, 3)) + rng.uniform(-.7, .7, 3)
        original, _ = validation._intersection_kind(a, b, _IDS, _IDS + 4)
        kind, calls, certified = validation._intersection_with_certificate(a, b, _IDS, _IDS + 4)
        assert kind == original
        if certified:
            assert kind == "disjoint" and calls == 0


@pytest.mark.parametrize("other,ids,expected", [
    (_UNIT.copy(), _IDS.copy(), "overlap"),
    (_UNIT * .2 + .1, _IDS + 4, "overlap"),
    (_UNIT * .5, [0, 4, 5, 6], "overlap"),
    (_UNIT * [1., 1., -1.], [4, 5, 6, 7], "nonconforming contact"),
    (np.array([[0., 0., 0.], [.4, 0., 0.], [0., .4, 0.], [0., 0., -.5]]),
     [0, 4, 5, 6], "nonconforming contact"),
    (np.array([[0., 0., 0.], [1., 0., 0.], [0., .4, 0.], [0., 0., -.5]]),
     [0, 1, 4, 5], "nonconforming contact"),
])
def test_bad_contacts_and_overlaps_never_receive_certificate(other, ids, expected):
    assert validation._plane_separation_kind(_UNIT, other, _IDS, ids) is None
    original, old_calls = validation._intersection_kind(_UNIT, other, _IDS, ids)
    kind, calls, certified = validation._intersection_with_certificate(_UNIT, other, _IDS, ids)
    assert kind == original == expected
    assert calls == old_calls and not certified


def test_near_contact_band_remains_with_original_lp():
    other = _UNIT * [1., 1., -1.]
    for delta in (-1e-9, 0., 1e-9):
        b = other + [0., 0., delta]
        assert validation._plane_separation_kind(_UNIT, b, _IDS, _IDS + 4) is None
        original, old_calls = validation._intersection_kind(_UNIT, b, _IDS, _IDS + 4)
        kind, calls, certified = validation._intersection_with_certificate(_UNIT, b, _IDS, _IDS + 4)
        assert kind == original and calls == old_calls and not certified


def test_nearly_coplanar_shared_face_is_not_certified():
    top, bottom = _UNIT.copy(), _UNIT.copy()
    top[3, 2], bottom[3, 2] = 1e-9, -1e-9
    assert validation._plane_separation_kind(top, bottom, _IDS, [0, 1, 2, 4]) is None


def test_bad_shared_coordinates_and_degenerate_inputs_not_certified():
    other = _UNIT * [1., 1., -1.] + [.001, 0., 0.]
    assert validation._plane_separation_kind(_UNIT, other, _IDS, [0, 1, 2, 4]) is None
    assert validation._plane_separation_kind(_UNIT, _UNIT, [0, 0, 2, 3], _IDS) is None
    assert validation._plane_separation_kind(np.zeros((4, 3)), np.zeros((4, 3)), _IDS, _IDS + 4) is None


def test_cube_pair_counts_and_no_lp_solver_calls(monkeypatch):
    vertices = np.array(list(product([0., 1.], repeat=3)))
    index = {tuple(point): i for i, point in enumerate(vertices)}
    cells = []
    for axes in permutations(range(3)):
        point = np.zeros(3)
        cell = [index[tuple(point)]]
        for axis in axes:
            point[axis] = 1.
            cell.append(index[tuple(point)])
        xyz = vertices[cell]
        if np.linalg.det((xyz[1:] - xyz[0]).T) < 0:
            cell[0], cell[1] = cell[1], cell[0]
        cells.append(cell)
    tets = np.array(cells)
    # First compare every pair with the unmodified LP before forbidding calls.
    for i, j in combinations(range(6), 2):
        assert validation._intersection_kind(vertices[tets[i]], vertices[tets[j]], tets[i], tets[j])[0] == "conforming"
    def unexpected_solver_call(*args, **kwargs):
        pytest.fail("a certified cube pair unnecessarily invoked linprog")
    monkeypatch.setattr(validation, "linprog", unexpected_solver_call)
    result = validation.validate_mesh(vertices, tets, _boundary(tets))
    assert result["valid"], result["errors"]
    assert result["checked_pairs"] == result["certified_pairs"] == 15
    assert result["lp_calls"] == 0


def test_uncertified_pairs_report_real_lp_calls(monkeypatch):
    vertices = np.vstack((_UNIT, _UNIT * .2 + .1))
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    actual_calls = 0
    original_linprog = validation.linprog
    def count_linprog(*args, **kwargs):
        nonlocal actual_calls
        actual_calls += 1
        return original_linprog(*args, **kwargs)
    monkeypatch.setattr(validation, "linprog", count_linprog)
    result = validation.validate_mesh(vertices, tets, _boundary(tets))
    assert not result["valid"] and result["overlap_count"] == 1
    assert result["checked_pairs"] == 1 and result["certified_pairs"] == 0
    assert result["lp_calls"] == actual_calls == 1


def test_disabled_pair_checks_have_zero_certificate_count():
    tets = np.array([[0, 1, 2, 3]])
    report = validation.validate_mesh(_UNIT, tets, _boundary(tets), check_overlap=False)
    assert report["valid"]
    assert report["certified_pairs"] == report["checked_pairs"] == report["lp_calls"] == 0
