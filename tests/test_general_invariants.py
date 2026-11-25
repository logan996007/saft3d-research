"""Front-chain orientation and legal cavity-shell merger regressions."""

from collections import defaultdict

import numpy as np
import pytest

from saft3d.front_geometry import FrontGeometry
from saft3d.general_mesher import AdvancingFrontConfig, StalledFront, _updated_front, grow_general_mesh
from saft3d.general_surfaces import make_general_surface
from saft3d.geometry import signed_volume, tet_faces, tet_overlap
from saft3d.mesher import face_key, orientation
from saft3d.surface_validation import _check_vertex_links
from saft3d.surfaces import cube


def _front(faces):
    return {face_key(face): tuple(map(int, face)) for face in faces}


def _edges(front):
    edges = defaultdict(list)
    for a, b, c in front.values():
        for x, y in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((x, y)))].append(1 if x < y else -1)
    return edges


def _volume(vertices, front):
    p = vertices[np.asarray(list(front.values()))]
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6)


def test_full_tet_consumption_and_orientation_rejection_are_transactional():
    front = _front(tet_faces((0, 1, 2, 3)))
    before = front.copy()
    assert _updated_front(front, (0, 1, 2, 3)) == {}
    assert _updated_front(front, (0, 2, 1, 3)) is None
    assert front == before


def test_new_side_faces_reverse_tetrahedron_outward_orientation():
    front = _front(tet_faces((0, 1, 2, 3)))
    indices = (0, 1, 2, 4)
    updated = _updated_front(front, indices)
    assert updated is not None
    assert face_key((0, 2, 1)) not in updated
    for face in tet_faces(indices):
        if 4 in face:
            assert orientation(updated[face_key(face)]) == -orientation(face)
    assert all(sum(signs) == 0 for signs in _edges(updated).values())


def test_one_component_can_close_without_erasing_another():
    remaining = _front(tet_faces((4, 5, 6, 7)))
    front = _front(tet_faces((0, 1, 2, 3))) | remaining
    assert _updated_front(front, (0, 1, 2, 3)) == remaining


def test_tetrahedron_fan_closes_with_consistent_orientation():
    boundary = tet_faces((0, 1, 2, 3))
    front = _front(boundary)
    for a, b, c in boundary:
        front = _updated_front(front, (a, c, b, 4))
        assert front is not None
        assert all(sum(signs) == 0 for signs in _edges(front).values())
    assert front == {}


def test_hollow_shell_bridge_requires_a_multicycle_intermediate_vertex():
    vertices, faces, _ = make_general_surface("hollow_cube")
    vertices = (vertices - vertices.min(axis=0)) / np.ptp(vertices, axis=0).max()
    front = _front(faces)
    # Grow from an outer bottom triangle to one lower corner of the inner shell.
    # The tet stays below the cavity and touches it only at existing vertex 8.
    indices = (0, 2, 3, 8)
    tet = vertices[list(indices)]
    original_geometry = FrontGeometry(vertices, faces)
    assert original_geometry.valid_candidate(tet, indices)
    updated = _updated_front(front, indices)
    assert updated is not None
    with pytest.raises(ValueError, match="disconnected cycle"):
        _check_vertex_links(np.asarray(list(updated.values())))
    assert all(sum(signs) == 0 for signs in _edges(updated).values())
    assert _volume(vertices, updated) == pytest.approx(_volume(vertices, front) - signed_volume(tet))
    remaining_geometry = FrontGeometry(vertices, np.asarray(list(updated.values())))
    assert original_geometry.contains(tet.mean(axis=0))
    assert not remaining_geometry.contains(tet.mean(axis=0))
    assert not remaining_geometry.contains([.5, .5, .5])  # the original cavity
    assert remaining_geometry.contains([.9, .9, .9])


def test_legal_inner_edge_contact_can_leave_four_balanced_front_faces():
    vertices, faces, _ = make_general_surface("hollow_cube")
    vertices = (vertices - vertices.min(axis=0)) / np.ptp(vertices, axis=0).max()
    first = (0, 2, 3, 8)
    second = (0, 2, 8, 9)
    front = _updated_front(_front(faces), first)
    geometry = FrontGeometry(vertices, np.asarray(list(front.values())))
    assert geometry.valid_candidate(vertices[list(second)], second)
    assert not tet_overlap(vertices[list(first)], vertices[list(second)])
    updated = _updated_front(front, second)
    assert updated is not None
    edges = _edges(updated)
    assert len(edges[(8, 9)]) == 4
    assert all(sum(signs) == 0 for signs in edges.values())
    assert _volume(vertices, updated) == pytest.approx(
        _volume(vertices, _front(faces))
        - signed_volume(vertices[list(first)]) - signed_volume(vertices[list(second)]))


@pytest.mark.parametrize("max_new,max_tets,expected_new,expected_tets", [
    (2, 100, 2, 2),  # Both separated components admit a candidate in one batch.
    (1, 100, 1, 1),  # The second new point must not overrun a mid-batch limit.
    (2, 1, 1, 1),    # Nor may the second tetrahedron overrun max_tets.
    (0, 100, 0, 2),  # No-new-point mode still considers existing-vertex candidates.
])
def test_resource_caps_are_strict_inside_a_batch(max_new, max_tets, expected_new, expected_tets):
    vertices, faces = cube()
    separated_vertices = np.vstack((vertices, vertices + [5., 0., 0.]))
    separated_faces = np.vstack((faces, faces + len(vertices)))
    config = AdvancingFrontConfig(backend="cpu", size=1., batch_size=32,
                                  max_new_vertices=max_new, max_tets=max_tets,
                                  max_iterations=1)
    # One iteration intentionally returns an explicitly incomplete diagnostic.
    with pytest.raises(StalledFront) as caught:
        grow_general_mesh(separated_vertices, separated_faces, config)
    partial = caught.value
    assert partial.report["new_vertices"] == expected_new
    assert len(partial.vertices) == len(separated_vertices) + expected_new
    assert len(partial.tetrahedra) == expected_tets
    assert partial.report["new_vertices"] <= max_new
    assert len(partial.tetrahedra) <= max_tets
    assert partial.report["remaining_front_faces"] > 0
