"""Analytical volume, embeddedness and non-star checks for surface fixtures."""

from collections import defaultdict

import numpy as np
import pytest

from saft3d.general_surfaces import GENERAL_SURFACE_NAMES, make_general_surface
from saft3d.mesher import UnsupportedDomain, visibility_kernel
from saft3d.surface_validation import check_embedded_surface


def _signed_surface_volume(vertices, faces):
    p = vertices[faces] - vertices.mean(axis=0)
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6)


@pytest.mark.parametrize("name", GENERAL_SURFACE_NAMES)
def test_surface_is_closed_oriented_and_embedded(name):
    vertices, faces, metadata = make_general_surface(name)
    assert vertices.shape[1] == faces.shape[1] == 3
    assert len(np.unique(vertices, axis=0)) == len(vertices)
    assert len({tuple(sorted(face)) for face in faces}) == len(faces)
    edges = defaultdict(list)
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            edges[tuple(sorted((a, b)))].append(1 if a < b else -1)
    assert all(len(orientations) == 2 and sum(orientations) == 0
               for orientations in edges.values())
    normalized = (vertices - vertices.min(axis=0)) / np.ptp(vertices, axis=0).max()
    assert check_embedded_surface(normalized, faces) is None
    assert _signed_surface_volume(vertices, faces) == pytest.approx(metadata["analytic_volume"], rel=1e-12)


@pytest.mark.parametrize("name", GENERAL_SURFACE_NAMES)
def test_no_positive_radius_visibility_kernel(name):
    vertices, faces, metadata = make_general_surface(name)
    assert metadata["expected_kernel"] is False
    normalized = (vertices - vertices.min(axis=0)) / np.ptp(vertices, axis=0).max()
    with pytest.raises(UnsupportedDomain, match="visibility kernel"):
        visibility_kernel(normalized, faces, 1e-10)


@pytest.mark.parametrize("name,n,face_count", [("u_prism", 8, 28), ("slender_s", 12, 44)])
def test_ear_clipped_caps_are_positive_and_cover_polygon(name, n, face_count):
    vertices, faces, metadata = make_general_surface(name)
    assert len(vertices) == 2 * n
    assert len(faces) == face_count
    bottom, top = faces[:n - 2], faces[n - 2:2 * (n - 2)]
    for cap, sign in ((bottom, -1), (top, 1)):
        p = vertices[cap]
        normal_z = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2]
        assert np.all(sign * normal_z > 0)
        area = np.abs(normal_z).sum() / 2
        assert area * metadata["extrusion_height"] == pytest.approx(metadata["analytic_volume"])


def test_cavity_shell_orientation_and_seed():
    vertices, faces, metadata = make_general_surface("hollow_cube")
    assert metadata["boundary_components"] == 2
    assert metadata["hole_seeds"] == [[0., 0., 0.]]
    assert _signed_surface_volume(vertices, faces[:12]) == pytest.approx(8.)
    assert _signed_surface_volume(vertices, faces[12:]) == pytest.approx(-.8 ** 3)
    center = np.asarray(metadata["hole_seeds"][0])
    p = vertices[faces[12:]]
    normals = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    assert np.all(np.einsum("ij,ij->i", normals, center - p[:, 0]) > 0)


def test_deterministic_generation_and_aliases():
    for name in GENERAL_SURFACE_NAMES:
        first = make_general_surface(name)
        second = make_general_surface(name)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        assert first[2] == second[2]
    assert make_general_surface("U-prism")[2]["name"] == "u_prism"
    with pytest.raises(ValueError, match="unknown general surface"):
        make_general_surface("unknown")
