from types import SimpleNamespace

import numpy as np
import pytest

from saft3d.surface_validation import check_embedded_surface
from saft3d.surfaces import cube, sphere


TET_FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])


def normalize(vertices):
    vertices = np.asarray(vertices, dtype=float)
    return (vertices - vertices.min(axis=0)) / np.ptp(vertices, axis=0).max()


@pytest.mark.parametrize("shape", [cube, lambda: sphere(0), lambda: sphere(1),
                                   lambda: sphere(2), lambda: sphere(1, True)])
def test_embedded_cube_and_spheres(shape):
    vertices, faces = shape()
    assert check_embedded_surface(normalize(vertices), faces) is None


def test_two_shells_sharing_only_one_vertex_rejected():
    # Each individual tetrahedral shell is closed and edge manifold.  Their
    # common vertex has a link consisting of two disconnected three-cycles.
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                         [-1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    second = np.array([0, 4, 6, 5])[TET_FACES]
    faces = np.vstack((TET_FACES, second))
    with pytest.raises(ValueError, match="vertex 0.*non-manifold.*disconnected cycle"):
        check_embedded_surface(normalize(vertices), faces)


@pytest.mark.parametrize("offset", [0.0, 2.0e-11])
def test_duplicate_and_near_coincident_coordinates_rejected(offset):
    vertices, faces = cube()
    vertices = normalize(vertices)
    vertices = np.vstack((vertices, vertices[0] + [offset, 0., 0.]))
    with pytest.raises(ValueError, match="duplicate or near-coincident"):
        check_embedded_surface(vertices, faces)


def crossing_shells():
    unit = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    vertices = np.vstack((unit, unit + [.2, .2, .2]))
    return normalize(vertices), np.vstack((TET_FACES, TET_FACES + 4))


def test_proper_crossing_surface_rejected():
    # Every vertex has a valid link, so this must reach geometric intersection
    # checks.  The two shells cross transversely, rather than just touch.
    vertices, faces = crossing_shells()
    with pytest.raises(ValueError, match="self-intersect|nonconforming contact"):
        check_embedded_surface(vertices, faces)


def test_nonconforming_coplanar_surface_contact_rejected():
    unit = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    below = np.array([[.1, .1, 0.], [.4, .1, 0.], [.1, .4, 0.], [.1, .1, -.4]])
    vertices = normalize(np.vstack((unit, below)))
    faces = np.vstack((TET_FACES, TET_FACES[:, ::-1] + 4))
    with pytest.raises(ValueError, match="self-intersect|nonconforming contact"):
        check_embedded_surface(vertices, faces)


def test_coplanar_overlap_beyond_a_shared_edge_rejected():
    # A flattened tetrahedral shell has valid combinatorial vertex links but
    # overlapping coplanar triangles; their indexed shared edges are insufficient.
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [.2, .2, 0.]])
    with pytest.raises(ValueError, match="outside their shared indexed edge"):
        check_embedded_surface(vertices, TET_FACES)


def test_index_order_does_not_change_embeddedness():
    vertices, faces = cube()
    order = np.array([5, 2, 7, 0, 6, 3, 1, 4])
    inverse = np.argsort(order)
    assert check_embedded_surface(normalize(vertices[order]), inverse[faces[::-1]]) is None


def test_triangle_limit_is_explicit():
    vertices, _ = cube()
    faces = np.tile([0, 1, 2], (2049, 1))
    with pytest.raises(ValueError, match="limited to 2048 triangles"):
        check_embedded_surface(normalize(vertices), faces)


def test_lp_failure_fails_closed(monkeypatch):
    vertices, faces = crossing_shells()
    monkeypatch.setattr(
        "saft3d.surface_validation.linprog",
        lambda *args, **kwargs: SimpleNamespace(success=False, status=4, message="numerical difficulty"),
    )
    with pytest.raises(ValueError, match="intersection LP failed.*numerical difficulty"):
        check_embedded_surface(vertices, faces)


def test_infeasible_lp_on_shared_simplex_fails_closed(monkeypatch):
    vertices, faces = cube()
    # Coplanar triangles of each cube face share an edge, so infeasibility is
    # inconsistent with their known shared simplex and cannot be accepted.
    monkeypatch.setattr(
        "saft3d.surface_validation.linprog",
        lambda *args, **kwargs: SimpleNamespace(success=False, status=2, message="infeasible"),
    )
    with pytest.raises(ValueError, match="intersection LP failed"):
        check_embedded_surface(normalize(vertices), faces)
