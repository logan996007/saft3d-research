"""Geometry regressions that exercise conformity and 3D intersection cases."""

import unittest

import numpy as np

from saft3d.geometry import (
    point_in_tet,
    signed_volume,
    tet_faces,
    tet_overlap,
    tet_quality,
    triangle_intersects_tet_interior,
)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.tet = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

    def test_signed_volume_and_inversion(self):
        self.assertAlmostEqual(signed_volume(self.tet), 1. / 6.)
        inverted = self.tet[[0, 2, 1, 3]]
        self.assertAlmostEqual(signed_volume(inverted), -1. / 6.)
        self.assertEqual(tet_quality(inverted), 0.)
        self.assertEqual(point_in_tet([.1, .1, .1], inverted), "inside")

    def test_regular_quality_and_sliver(self):
        regular = np.array([[0., 0., 0.], [1., 0., 0.], [.5, np.sqrt(3.) / 2., 0.],
                            [.5, np.sqrt(3.) / 6., np.sqrt(2. / 3.)]])
        self.assertAlmostEqual(tet_quality(regular), 1.)
        self.assertAlmostEqual(tet_quality(regular * 1e-8), 1.)
        sliver = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 1e-9]])
        self.assertGreater(signed_volume(sliver), 0.)
        self.assertLess(tet_quality(sliver), 1e-5)

    def test_faces_point_outward(self):
        center = self.tet.mean(axis=0)
        for ids in tet_faces((0, 1, 2, 3)):
            face = self.tet[list(ids)]
            normal = np.cross(face[1] - face[0], face[2] - face[0])
            self.assertLess(np.dot(normal, center - face[0]), 0.)

    def test_point_classification(self):
        self.assertEqual(point_in_tet([.1, .2, .3], self.tet), "inside")
        self.assertEqual(point_in_tet([.1, .2, 0.], self.tet), "boundary")
        self.assertEqual(point_in_tet([1., 1., 1.], self.tet), "outside")
        self.assertEqual(point_in_tet([0., 0., 0.], np.zeros((4, 3))), "outside")

    def test_identical_contained_and_separated(self):
        self.assertTrue(tet_overlap(self.tet, self.tet))
        self.assertTrue(tet_overlap(self.tet, self.tet * .1 + .1))
        self.assertFalse(tet_overlap(self.tet, self.tet + 2.))

    def test_shared_face_edge_vertex_are_legal(self):
        across_face = self.tet.copy()
        across_face[3, 2] = -1.
        self.assertFalse(tet_overlap(self.tet, across_face))
        across_edge = self.tet.copy()
        across_edge[2, 1] = -1.
        across_edge[3, 2] = -1.
        self.assertFalse(tet_overlap(self.tet, across_edge))
        self.assertFalse(tet_overlap(self.tet, -self.tet))

    def test_crossing_without_contained_vertices(self):
        # Two dual regular tetrahedra: their intersection is an octahedron.
        # A vertex-only intersection test misses every collision here.
        a = np.array([[1., 1., 1.], [1., -1., -1.], [-1., 1., -1.], [-1., -1., 1.]])
        b = -a
        for vertex in a:
            self.assertEqual(point_in_tet(vertex, b), "outside")
        for vertex in b:
            self.assertEqual(point_in_tet(vertex, a), "outside")
        self.assertTrue(tet_overlap(a, b))

    def test_overlap_is_translation_invariant(self):
        shift = np.array([1e6, -1e6, 1e6])
        other = self.tet + .1
        self.assertEqual(tet_overlap(self.tet, other), tet_overlap(self.tet + shift, other + shift))

    def test_triangle_boundary_contacts_are_legal(self):
        for face in tet_faces((0, 1, 2, 3)):
            triangle = self.tet[list(face)]
            self.assertFalse(triangle_intersects_tet_interior(triangle, self.tet))
            self.assertFalse(triangle_intersects_tet_interior(triangle, self.tet, tol=0.))
        large_coplanar = np.array([[-2., -2., 0.], [3., -2., 0.], [-2., 3., 0.]])
        self.assertFalse(triangle_intersects_tet_interior(large_coplanar, self.tet))

    def test_triangle_crosses_without_contained_vertices(self):
        triangle = np.array([[-2., -2., .2], [3., -2., .2], [-2., 3., .2]])
        self.assertTrue(all(point_in_tet(v, self.tet) == "outside" for v in triangle))
        self.assertTrue(triangle_intersects_tet_interior(triangle, self.tet))
        self.assertTrue(triangle_intersects_tet_interior(triangle, self.tet[[0, 2, 1, 3]]))

    def test_inside_triangle_and_outside_triangle(self):
        triangle = np.array([[.1, .1, .1], [.3, .1, .1], [.1, .3, .1]])
        self.assertTrue(triangle_intersects_tet_interior(triangle, self.tet))
        self.assertFalse(triangle_intersects_tet_interior(triangle + 2., self.tet))

    def test_tolerance_is_absolute_distance(self):
        a = self.tet.copy()
        b = self.tet.copy()
        b[3, 2] = -1.
        b[:, 2] += 1e-7
        self.assertTrue(tet_overlap(a, b, tol=1e-9))
        self.assertFalse(tet_overlap(a, b, tol=1e-5))
        self.assertTrue(tet_overlap(a * 1e3, b * 1e3, tol=1e-5))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            signed_volume(np.zeros((3, 3)))
        with self.assertRaises(ValueError):
            tet_faces((0, 1, 1, 2))
        with self.assertRaises(ValueError):
            tet_overlap(self.tet, self.tet, tol=-1.)
        with self.assertRaises(ValueError):
            point_in_tet([np.nan, 0., 0.], self.tet)


if __name__ == "__main__":
    unittest.main()
