import json
from collections import Counter
from itertools import permutations, product

import numpy as np
import pytest

from saft3d.validation import validate_mesh


FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))


def boundary_of(tets):
    oriented = [tuple(int(t[j]) for j in f) for t in tets for f in FACES]
    counts = Counter(tuple(sorted(f)) for f in oriented)
    return np.asarray([f for f in oriented if counts[tuple(sorted(f))] == 1], dtype=int)


def positive_tets(vertices, tets):
    tets = np.array(tets, dtype=int)
    for t in tets:
        x = vertices[t]
        if np.linalg.det((x[1:] - x[0]).T) < 0:
            t[0], t[1] = t[1], t[0]
    return tets


@pytest.fixture
def single_tet():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    tets = np.array([[0, 1, 2, 3]])
    return vertices, tets, boundary_of(tets)


def test_positive_single_tet_and_json(single_tet):
    result = validate_mesh(*single_tet)
    assert result["valid"], result["errors"]
    assert result["positive_volume_sum"] == pytest.approx(1 / 6)
    assert result["signed_boundary_volume"] == pytest.approx(1 / 6)
    assert result["boundary_faces_match"]
    assert result["quality"]["min"] > 0
    json.dumps(result, allow_nan=False)


def test_inverted_tet_fails(single_tet):
    vertices, tets, boundary = single_tet
    tets = tets[:, [1, 0, 2, 3]]
    result = validate_mesh(vertices, tets, boundary)
    assert not result["valid"]
    assert result["inverted_tet_count"] == 1


def test_wrong_boundary_fails(single_tet):
    vertices, tets, boundary = single_tet
    result = validate_mesh(vertices, tets, boundary[:-1])
    assert not result["valid"]
    assert not result["boundary_faces_match"]
    assert result["missing_boundary_face_count"] == 1
    assert not result["boundary_closed_oriented"]


def test_reversed_boundary_orientation_fails(single_tet):
    vertices, tets, boundary = single_tet
    result = validate_mesh(vertices, tets, boundary[:, ::-1])
    assert not result["valid"]
    assert result["boundary_faces_match"]
    assert result["boundary_orientation_error_count"] == 4
    assert result["signed_boundary_volume"] < 0


def test_two_tets_with_conforming_shared_face():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                         [0., 0., 1.], [0., 0., -1.]])
    tets = positive_tets(vertices, [[0, 1, 2, 3], [0, 1, 2, 4]])
    result = validate_mesh(vertices, tets, boundary_of(tets))
    assert result["valid"], result["errors"]
    assert result["internal_face_count"] == 1
    assert result["checked_pairs"] == 1
    assert result["overlap_count"] == 0
    assert result["nonconforming_contact_count"] == 0


def test_six_tet_cube_accepts_conforming_faces_and_edges():
    vertices = np.array(list(product([0., 1.], repeat=3)))
    index = {tuple(point): i for i, point in enumerate(vertices)}
    cells = []
    for axes in permutations(range(3)):
        point = np.zeros(3)
        cell = [index[tuple(point)]]
        for axis in axes:
            point[axis] = 1
            cell.append(index[tuple(point)])
        cells.append(cell)
    tets = positive_tets(vertices, cells)
    result = validate_mesh(vertices, tets, boundary_of(tets))
    assert result["valid"], result["errors"]
    assert result["positive_volume_sum"] == pytest.approx(1.0)
    assert result["boundary_face_count"] == 12
    assert result["checked_pairs"] == 15
    assert result["nonconforming_contact_count"] == 0


def test_equilateral_tet_quality_is_one():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.],
                         [.5, np.sqrt(3) / 2, 0.],
                         [.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]])
    tets = np.array([[0, 1, 2, 3]])
    result = validate_mesh(vertices, tets, boundary_of(tets))
    assert result["valid"]
    assert result["quality"]["min"] == pytest.approx(1.0)


def test_contained_tet_overlap_is_detected():
    vertices = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.],
                         [.1, .1, .1], [.3, .1, .1], [.1, .3, .1], [.1, .1, .3]])
    tets = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    result = validate_mesh(vertices, tets, boundary_of(tets))
    # Algebraic signed-volume agreement alone cannot discover this containment.
    assert result["relative_volume_error"] < 1e-10
    assert result["boundary_faces_match"]
    assert not result["valid"]
    assert result["overlap_pairs"] == [[0, 1]]


def test_equal_volume_overlap_and_hole_do_not_pass():
    # Desired domain: disjoint unit tetrahedra at x=0 and x=3.  Replace the
    # second by an equal-volume tet overlapping the first; total volume agrees.
    unit = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    vertices = np.vstack((unit, unit + [3., 0., 0.], unit + [.2, .1, .1]))
    target = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    bad = np.array([[0, 1, 2, 3], [8, 9, 10, 11]])
    result = validate_mesh(vertices, bad, boundary_of(target))
    assert result["relative_volume_error"] < 1e-10
    assert not result["valid"]
    assert not result["boundary_faces_match"]
    assert result["overlap_count"] == 1


def test_nonconforming_face_contact_detected_without_volume_overlap():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [.1, .1, 0.], [.4, .1, 0.], [.1, .4, 0.], [.1, .1, -.5]])
    tets = positive_tets(vertices, [[0, 1, 2, 3], [4, 5, 6, 7]])
    result = validate_mesh(vertices, tets, boundary_of(tets))
    assert not result["valid"]
    assert result["overlap_count"] == 0
    assert result["nonconforming_contact_pairs"] == [[0, 1]]


def test_contact_beyond_a_shared_vertex_is_detected():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [.4, 0., 0.], [0., .4, 0.], [0., 0., -.5]])
    tets = positive_tets(vertices, [[0, 1, 2, 3], [0, 4, 5, 6]])
    result = validate_mesh(vertices, tets, boundary_of(tets))
    assert result["overlap_count"] == 0
    assert result["nonconforming_contact_count"] == 1


def test_duplicate_tet_detected(single_tet):
    vertices, tets, boundary = single_tet
    result = validate_mesh(vertices, np.vstack((tets, tets)), boundary)
    assert not result["valid"]
    assert result["duplicate_tet_count"] == 1
    assert result["overlap_count"] == 1
    assert result["internal_orientation_error_count"] == 4


def test_unused_vertices_are_informational(single_tet):
    vertices, tets, boundary = single_tet
    result = validate_mesh(np.vstack((vertices, [1e8, 1e8, 1e8])), tets, boundary)
    assert result["valid"], result["errors"]
    assert result["unused_vertex_indices"] == [4]


def test_coplanar_tet_fails(single_tet):
    vertices, tets, boundary = single_tet
    vertices[3] = [.2, .2, 0.]
    result = validate_mesh(vertices, tets, boundary)
    assert not result["valid"]
    assert result["degenerate_tet_count"] == 1


@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_uniform_scale_and_translation(single_tet, scale):
    vertices, tets, boundary = single_tet
    vertices = vertices * scale + np.array([123., -45., 6.]) * scale
    result = validate_mesh(vertices, tets, boundary)
    assert result["valid"], result["errors"]
    assert result["relative_volume_error"] < 1e-10


def test_fractional_and_out_of_range_indices_rejected(single_tet):
    vertices, tets, boundary = single_tet
    result = validate_mesh(vertices, [[0, 1, 2, 3.1]], boundary)
    assert not result["valid"]
    result = validate_mesh(vertices, [[0, 1, 2, 99]], boundary)
    assert not result["valid"]


def test_disabled_intersection_checks_are_explicit(single_tet):
    result = validate_mesh(*single_tet, check_overlap=False)
    assert result["valid"]
    assert not result["geometric_intersections_checked"]
    assert result["checked_pairs"] == 0
