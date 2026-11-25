"""Fixed small cavities with analytically different best triangulations."""

from collections import defaultdict

import numpy as np
import pytest

from saft3d.geometry import signed_volume, tet_faces
from saft3d.topology_optimization import improve_topology
from saft3d.validation import validate_mesh


def _orient(vertices, tets):
    tets = np.asarray(tets, dtype=np.int64).copy()
    for tet in tets:
        if signed_volume(vertices[tet]) < 0:
            tet[0], tet[1] = tet[1], tet[0]
    return tets


def _boundary(tets):
    incidence = defaultdict(list)
    for tet in tets:
        for face in tet_faces(tet):
            incidence[tuple(sorted(face))].append(face)
    return np.asarray([faces[0] for faces in incidence.values() if len(faces) == 1])


def _bipyramid(height, count):
    vertices = np.array([[1., 0., 0.], [-.5, np.sqrt(3.) / 2, 0.],
                         [-.5, -np.sqrt(3.) / 2, 0.], [0., 0., height], [0., 0., -height]])
    tets = ([(0, 1, 2, 3), (0, 2, 1, 4)] if count == 2
            else [(3, 4, 0, 1), (3, 4, 1, 2), (3, 4, 2, 0)])
    tets = _orient(vertices, tets)
    return vertices, tets, _boundary(tets)


@pytest.mark.parametrize("height,old_count,new_count,kind", [
    (.2, 2, 3, "2_to_3"), (1., 3, 2, "3_to_2"),
])
def test_real_flips_improve_quality_and_preserve_exact_boundary(height, old_count, new_count, kind):
    vertices, tets, boundary = _bipyramid(height, old_count)
    saved_vertices, saved_tets, saved_boundary = vertices.copy(), tets.copy(), boundary.copy()
    output, report = improve_topology(vertices, tets, boundary)
    assert len(output) == new_count
    assert report["flips"][kind] == 1
    assert report["accepted_flips"] == 1
    assert report["after"]["min"] > report["before"]["min"] + .05
    assert report["boundary_preserved"]
    np.testing.assert_array_equal(vertices, saved_vertices)
    np.testing.assert_array_equal(tets, saved_tets)
    np.testing.assert_array_equal(boundary, saved_boundary)
    validation = validate_mesh(vertices, output, boundary)
    assert validation["valid"], validation["errors"]
    assert validation["boundary_faces_match"]
    assert validation["relative_volume_error"] < 1e-12


def test_already_better_connectivity_stays_unchanged():
    vertices, tets, boundary = _bipyramid(1., 2)
    output, report = improve_topology(vertices, tets, boundary)
    np.testing.assert_array_equal(output, tets)
    assert report["accepted_flips"] == 0
    assert report["before"] == report["after"]


def test_nonconvex_bipyramid_cannot_be_flipped_across_its_exterior():
    vertices, tets, _ = _bipyramid(.2, 2)
    vertices[3:, 0] = 2.0  # The proposed apex-to-apex edge misses the shared triangle.
    tets = _orient(vertices, tets)
    boundary = _boundary(tets)
    initial = validate_mesh(vertices, tets, boundary)
    assert initial["valid"], initial["errors"]
    output, report = improve_topology(vertices, tets, boundary)
    np.testing.assert_array_equal(output, tets)
    assert report["accepted_flips"] == 0


def test_patch_replacement_remains_conforming_with_an_external_neighbor():
    vertices, tets, boundary = _bipyramid(.2, 2)
    shared = boundary[0]
    face = vertices[shared]
    normal = np.cross(face[1] - face[0], face[2] - face[0])
    normal /= np.linalg.norm(normal)
    extra = face.mean(axis=0) + .3 * normal
    vertices = np.vstack((vertices, extra))
    tets = _orient(vertices, np.vstack((tets, [*shared, 5])))
    boundary = _boundary(tets)
    initial = validate_mesh(vertices, tets, boundary)
    assert initial["valid"], initial["errors"]
    output, report = improve_topology(vertices, tets, boundary)
    checked = validate_mesh(vertices, output, boundary)
    assert checked["valid"], checked["errors"]
    assert checked["boundary_faces_match"]
    assert report["after"]["min"] >= report["before"]["min"]


def test_min_gain_and_pass_budget_can_disable_changes():
    vertices, tets, boundary = _bipyramid(.2, 2)
    for kwargs in ({"min_gain": 1.0}, {"max_passes": 0}):
        output, report = improve_topology(vertices, tets, boundary, **kwargs)
        np.testing.assert_array_equal(output, tets)
        assert report["accepted_flips"] == 0


def test_inverted_input_and_changed_prescribed_boundary_are_rejected():
    vertices, tets, boundary = _bipyramid(.2, 2)
    inverted = tets.copy()
    inverted[0, [0, 1]] = inverted[0, [1, 0]]
    with pytest.raises(ValueError, match="inverted or degenerate"):
        improve_topology(vertices, inverted, boundary)
    with pytest.raises(ValueError, match="prescribed oriented boundary"):
        improve_topology(vertices, tets, boundary[:, ::-1])
