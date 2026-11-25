"""Core geometric/topological invariants of the restricted front generator."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from saft3d.geometry import point_in_tet, signed_volume
from saft3d.mesher import MesherConfig, UnsupportedDomain, grow_mesh
from saft3d.surfaces import tetrahedron, torus
from saft3d.validation import validate_mesh
from saft3d.viewer import write_viewer


class MesherGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original, cls.boundary = tetrahedron()
        cls.vertices, cls.tets, cls.report = grow_mesh(
            cls.original, cls.boundary,
            MesherConfig(backend="cpu", growth_rounds=1, size=.6, batch_size=8),
        )

    def test_growth_and_closure_preserve_boundary_and_volume(self):
        self.assertGreater(self.report["growth_tets"], 0)
        self.assertGreater(self.report["closure_tets"], 0)
        self.assertEqual(self.report["remaining_front_faces"], 0)
        self.assertTrue(all(signed_volume(self.vertices[t]) > 0 for t in self.tets))
        # The independent LP validator also detects illegal contacts between
        # cells that share no indexed vertex or only part of a shared face.
        checked = validate_mesh(self.vertices, self.tets, self.boundary)
        self.assertTrue(checked["valid"], checked["errors"])
        self.assertTrue(checked["boundary_faces_match"])
        self.assertLess(checked["relative_volume_error"], 1e-12)

    def test_all_points_inside_convex_tetrahedron(self):
        self.assertTrue(all(point_in_tet(p, self.original) != "outside" for p in self.vertices))
        self.assertEqual(len(np.unique(self.vertices, axis=0)), len(self.vertices))
        anchor = np.asarray(self.report["kernel_point"])
        anchor_ids = np.flatnonzero(np.linalg.norm(self.vertices - anchor, axis=1) < 1e-12)
        self.assertEqual(len(anchor_ids), 1)

    def test_closed_non_star_domain_is_explicitly_rejected(self):
        vertices, faces = torus(n_major=6, n_minor=6)
        with self.assertRaises(UnsupportedDomain):
            grow_mesh(vertices, faces, MesherConfig(backend="cpu", growth_rounds=0))

    def test_reversed_surface_is_rejected(self):
        with self.assertRaises(ValueError):
            grow_mesh(self.original, self.boundary[:, ::-1], MesherConfig(backend="cpu"))

    def test_viewer_is_standalone_and_json_is_inert(self):
        malicious_string = "</script><script>alert('report')</script>"
        with tempfile.TemporaryDirectory() as folder:
            path = write_viewer(Path(folder) / "viewer.html", self.vertices, self.tets,
                                self.boundary, {"note": malicious_string})
            html = path.read_text(encoding="utf-8")
        self.assertIn('<canvas id="view"', html)
        self.assertIn('id="cut"', html)
        self.assertIn('id="explode"', html)
        self.assertNotIn('<script src=', html)
        self.assertNotIn(malicious_string, html)
        payload = html.split('<script type="application/json" id="mesh-data">', 1)[1].split('</script>', 1)[0]
        decoded = json.loads(payload)
        self.assertEqual(decoded["report"]["note"], malicious_string)
        self.assertEqual(len(decoded["tets"]), len(self.tets))


if __name__ == "__main__":
    unittest.main()
