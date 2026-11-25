from types import SimpleNamespace

import numpy as np
import pytest

from saft3d.front_geometry import FrontGeometry
from saft3d.surfaces import cube


TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


def unit_cube():
    vertices, faces = cube()
    return (vertices + 1) / 2, faces


def inward_candidate(vertices, face, depth=0.15):
    triangle = vertices[face]
    normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
    normal /= np.linalg.norm(normal)
    apex = triangle.mean(axis=0) - depth * normal
    return np.vstack((triangle[[0, 2, 1]], apex)), [int(face[0]), int(face[2]), int(face[1]), -1]


def test_cube_contains_excludes_boundary_and_outside():
    vertices, faces = unit_cube()
    front = FrontGeometry(vertices, faces)
    assert front.contains([0.5, 0.5, 0.5])
    assert front.contains([0.02, 0.3, 0.6])
    assert not front.contains([1.01, 0.5, 0.5])
    assert not front.contains([0., 0.5, 0.5])
    assert not front.contains([0., 0., 0.])


@pytest.mark.parametrize("face_index", [0, 1, 2, 5, 8, 11])
def test_inward_tet_with_legal_shared_face_and_edges(face_index):
    vertices, faces = unit_cube()
    front = FrontGeometry(vertices, faces)
    tet, ids = inward_candidate(vertices, faces[face_index])
    assert front.valid_candidate(tet, ids), front.stats
    assert front.stats["calls"] == 1
    assert front.stats["pair_checks"] > 1
    assert front.stats["rejections"] == 0


def test_outward_tet_rejected_even_with_positive_ordering():
    vertices, faces = unit_cube()
    front = FrontGeometry(vertices, faces)
    tet, ids = inward_candidate(vertices, faces[0], depth=-0.2)
    tet[[0, 1]] = tet[[1, 0]]
    ids[0], ids[1] = ids[1], ids[0]
    assert not front.valid_candidate(tet, ids)
    assert front.stats["last_rejection"]["reason"] == "centroid_outside_remaining_domain"


def test_crossing_front_rejected_when_centroid_remains_inside():
    vertices, faces = unit_cube()
    front = FrontGeometry(vertices, faces)
    tet, ids = inward_candidate(vertices, faces[0], depth=2.0)
    assert front.contains(tet.mean(axis=0))
    assert not front.valid_candidate(tet, ids)
    assert front.stats["last_rejection"]["reason"] == "front_intersection_or_nonconforming_contact"


def test_front_vertex_in_middle_of_candidate_face_is_rejected():
    vertices, faces = unit_cube()
    # Insert a vertex on the common diagonal of the first square side.  The
    # proposed unsplit face would cover existing smaller triangles and that vertex.
    first = faces[0]
    partner = next(i for i in range(1, len(faces))
                   if len(set(first).intersection(faces[i])) == 2
                   and np.all(vertices[faces[i], 2] == vertices[first[0], 2]))
    shared_edge = list(set(first).intersection(faces[partner]))
    center = vertices[shared_edge].mean(axis=0)
    new_id = len(vertices)
    vertices = np.vstack((vertices, center))
    refined = []
    for i, face in enumerate(faces):
        if i not in (0, partner):
            refined.append(face)
            continue
        # Replace each triangle by two, preserving its oriented boundary.
        for j in range(3):
            a, b, c = np.roll(face, -j)
            if {int(a), int(b)} == set(shared_edge):
                refined.extend([[a, new_id, c], [new_id, b, c]])
                break
    front = FrontGeometry(vertices, np.array(refined))
    tet, ids = inward_candidate(vertices, first)
    assert front.contains(tet.mean(axis=0))
    assert not front.valid_candidate(tet, ids)
    assert front.stats["last_rejection"]["reason"] == "front_intersection_or_nonconforming_contact"


def test_complete_tetrahedral_domain_with_existing_ids_is_legal():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    front = FrontGeometry(vertices, TET_FACES)
    assert front.valid_candidate(vertices, [0, 1, 2, 3]), front.stats
    assert front.stats["lp_calls"] == 0


def test_winding_cancels_in_cavity():
    outer_vertices, outer_faces = unit_cube()
    inner_vertices = outer_vertices * .4 + .3
    vertices = np.vstack((outer_vertices, inner_vertices))
    faces = np.vstack((outer_faces, outer_faces[:, ::-1] + 8))
    front = FrontGeometry(vertices, faces)
    assert front.contains([.1, .1, .1])
    assert not front.contains([.5, .5, .5])
    assert not front.contains([.3, .5, .5])
    assert not front.contains([1.1, .5, .5])


def test_detached_front_triangles_inside_candidate_are_rejected():
    outer = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    inner, inner_faces = unit_cube()
    inner = inner * .04 + .08
    vertices = np.vstack((outer, inner))
    faces = np.vstack((TET_FACES, inner_faces[:, ::-1] + 4))
    front = FrontGeometry(vertices, faces)
    # The candidate's centroid is in material, but it encloses the entire void.
    assert front.contains(outer.mean(axis=0))
    assert not front.valid_candidate(outer, [0, 1, 2, 3])
    assert front.stats["last_rejection"]["face_index"] >= 4
    assert front.stats["last_rejection"]["reason"] == "front_intersection_or_nonconforming_contact"


def test_new_vertex_cannot_duplicate_an_existing_front_vertex():
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    front = FrontGeometry(vertices, TET_FACES)
    assert not front.valid_candidate(vertices, [0, 1, 2, -1])
    assert front.stats["last_rejection"]["reason"] == "new_vertex_duplicates_existing_vertex"


def test_inverted_tet_and_wrong_vertex_id_rejected():
    vertices, faces = unit_cube()
    front = FrontGeometry(vertices, faces)
    tet, ids = inward_candidate(vertices, faces[0])
    tet[[0, 1]] = tet[[1, 0]]
    ids[0], ids[1] = ids[1], ids[0]
    assert not front.valid_candidate(tet, ids)
    assert front.stats["last_rejection"]["reason"] == "nonpositive_or_degenerate_tet"
    tet, ids = inward_candidate(vertices, faces[0])
    tet[0] += [.01, 0, 0]
    assert not front.valid_candidate(tet, ids)
    assert front.stats["last_rejection"]["reason"] == "vertex_id_coordinate_mismatch"


def test_lp_failure_rejects_candidate(monkeypatch):
    vertices, faces = unit_cube()
    front = FrontGeometry(vertices, faces)
    tet, ids = inward_candidate(vertices, faces[0], depth=2.)
    monkeypatch.setattr(
        "saft3d.front_geometry.linprog",
        lambda *args, **kwargs: SimpleNamespace(success=False, status=4, message="failed"),
    )
    assert not front.valid_candidate(tet, ids)
    assert front.stats["last_rejection"]["reason"] == "intersection_lp_failure"
