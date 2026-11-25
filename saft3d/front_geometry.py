"""Numerical local legality checks for general advancing-front tetrahedra.

No visibility kernel or tetrahedralization library is used.  The current front
must be a valid, closed, outward-oriented embedded surface; cavity shells point
into the excluded holes.  The caller remains responsible for transactional
front topology updates and conflicts between candidates in the same batch.

Triangle/tetrahedron intersection is checked independently with barycentric
linear programs.  The predicates are tolerance based, not exact arithmetic.
"""

from collections import Counter
from math import fsum

import numpy as np
from scipy.optimize import linprog


_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
_LP_OPTIONS = {
    "primal_feasibility_tolerance": 1.0e-10,
    "dual_feasibility_tolerance": 1.0e-10,
    "ipm_optimality_tolerance": 1.0e-12,
}


class FrontGeometry:
    """Immutable geometry snapshot of an outward-oriented active front."""

    def __init__(self, points, faces, tol=1.0e-10):
        self.tol = float(tol)
        if not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("front tolerance must be finite and nonnegative")
        self.points = np.array(points, dtype=float, copy=True)
        raw_faces = np.asarray(faces)
        if (self.points.ndim != 2 or self.points.shape[1] != 3
                or not np.isfinite(self.points).all()):
            raise ValueError("front points must be a finite Nx3 array")
        if raw_faces.ndim != 2 or raw_faces.shape[1] != 3 or not len(raw_faces):
            raise ValueError("front faces must be a nonempty Fx3 array")
        numeric = np.asarray(raw_faces, dtype=float)
        if (not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all()
                or np.any(numeric < 0) or np.any(numeric >= len(self.points))):
            raise ValueError("front faces contain invalid vertex indices")
        self.faces = numeric.astype(np.int64)
        if any(len(set(map(int, face))) != 3 for face in self.faces):
            raise ValueError("front triangle repeats a vertex index")
        self.triangles = self.points[self.faces]
        normals = np.cross(self.triangles[:, 1] - self.triangles[:, 0],
                           self.triangles[:, 2] - self.triangles[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        scales = np.max(np.ptp(self.triangles, axis=1), axis=1)
        if np.any(lengths <= np.maximum(self.tol * scales,
                                        64 * np.finfo(float).eps * scales**2)):
            raise ValueError("front contains a degenerate triangle")
        self.normals = normals / lengths[:, None]
        self.lows = self.triangles.min(axis=1)
        self.highs = self.triangles.max(axis=1)
        self.stats = {
            "calls": 0, "contains_calls": 0, "pair_checks": 0, "lp_calls": 0,
            "rejections": 0, "reasons": {}, "last_rejection": None,
        }

    def _reject(self, reason, **details):
        self.stats["rejections"] += 1
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
        self.stats["last_rejection"] = {"reason": reason, **details}
        return False

    def contains(self, point):
        """Whether a point is strictly inside the oriented remaining material.

        Summed solid angles cancel inside correctly oriented cavity shells.
        Points on the front within the distance tolerance are rejected.
        """
        self.stats["contains_calls"] += 1
        query = np.asarray(point, dtype=float)
        if query.shape != (3,) or not np.isfinite(query).all():
            return False
        plane_distance = np.einsum("ij,ij->i", query - self.triangles[:, 0], self.normals)
        close = np.flatnonzero(np.abs(plane_distance) <= self.tol)
        if len(close):
            triangles, normals = self.triangles[close], self.normals[close]
            on_triangle = np.ones(len(close), dtype=bool)
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = triangles[:, j] - triangles[:, i]
                signed = np.einsum(
                    "ij,ij->i", np.cross(edge, query - triangles[:, i]), normals,
                )
                on_triangle &= signed >= -self.tol * np.linalg.norm(edge, axis=1)
            if np.any(on_triangle):
                return False
        relative = self.triangles - query
        a, b, c = relative[:, 0], relative[:, 1], relative[:, 2]
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (la * lb * lc + np.einsum("ij,ij->i", a, b) * lc
                       + np.einsum("ij,ij->i", b, c) * la
                       + np.einsum("ij,ij->i", c, a) * lb)
        angles = 2.0 * np.arctan2(numerator, denominator)
        if not np.isfinite(angles).all():
            return False
        winding = fsum(map(float, angles)) / (4.0 * np.pi)
        return bool(0.5 < winding < 1.5)

    def _pair_is_conforming(self, tet, ids, tet_normals, tet_anchors, face_index):
        """Return (allowed, reason); all feasible contact must be indexed."""
        triangle = self.triangles[face_index]
        face_ids = self.faces[face_index]
        shared = set(map(int, ids)).intersection(map(int, face_ids))
        shared.discard(-1)
        tri_exclusive = np.array([int(v) not in shared for v in face_ids])
        tet_exclusive = np.array([int(v) not in shared for v in ids])

        if len(shared) == 3:
            # The complete front triangle is a tetrahedron face.  A domain
            # boundary is locally filled only on its inward side.
            opposite = tet[tet_exclusive][0]
            distance = float(np.dot(opposite - triangle[0], self.normals[face_index]))
            return (distance < -self.tol, "wrong_side_of_shared_face")

        # A front triangle wholly outside a tetrahedron half-space is disjoint.
        # If only its non-shared vertices are outside and its shared vertices
        # lie on the separating plane, the contact is confined to that simplex.
        plane_tol = max(self.tol, 64.0 * np.finfo(float).eps)
        distances = np.einsum("fij,fj->fi", triangle[None, :, :] - tet_anchors[:, None, :],
                              tet_normals)
        if np.any(np.all(distances > plane_tol, axis=1)):
            return True, "disjoint"
        if shared:
            for row in distances:
                if (np.all(row[tri_exclusive] > plane_tol)
                        and np.all(np.abs(row[~tri_exclusive]) <= plane_tol)):
                    return True, "shared_simplex"

        # Conversely, intersection with the front triangle's plane is confined
        # to shared IDs if all other tet vertices lie strictly on one side.
        side = (tet - triangle[0]) @ self.normals[face_index]
        exclusive_side = side[tet_exclusive]
        if (np.all(exclusive_side > plane_tol) or np.all(exclusive_side < -plane_tol)):
            if not shared or np.all(np.abs(side[~tet_exclusive]) <= plane_tol):
                return True, "shared_simplex" if shared else "disjoint"

        combined = np.concatenate((triangle, tet))
        origin = combined.min(axis=0)
        scale = float(np.max(np.ptp(combined, axis=0)))
        equality = np.zeros((5, 7))
        equality[0, :3] = 1.0
        equality[1, 3:] = 1.0
        equality[2:, :3] = ((triangle - origin) / scale).T
        equality[2:, 3:] = -((tet - origin) / scale).T
        rhs = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
        objective = -np.concatenate((tri_exclusive, tet_exclusive)).astype(float) if shared else np.zeros(7)
        self.stats["lp_calls"] += 1
        try:
            result = linprog(
                objective, A_eq=equality, b_eq=rhs, bounds=(0.0, None),
                method="highs", options=_LP_OPTIONS,
            )
        except Exception:
            return False, "intersection_lp_failure"
        if result.status == 2 and not shared:
            return True, "disjoint"
        if not result.success:
            return False, "intersection_lp_failure"
        solution = np.asarray(result.x, dtype=float)
        if (solution.shape != (7,) or not np.isfinite(solution).all()
                or not np.isfinite(result.fun) or np.min(solution) < -1.0e-9
                or np.max(np.abs(equality @ solution - rhs)) > 1.0e-9):
            return False, "intersection_lp_failure"
        bary_tol = max(1.0e-9, 8.0 * self.tol / scale)
        if not shared or -float(result.fun) > bary_tol:
            return False, "front_intersection_or_nonconforming_contact"
        return True, "shared_simplex"

    def valid_candidate(self, tet_coords, vertex_ids):
        """Check one candidate relative to this snapshot of the unfilled domain.

        Existing vertex IDs must match their coordinates.  At most one new
        vertex is permitted and is identified by -1.  All front triangles,
        including detached triangles entirely inside the candidate, are checked.
        """
        self.stats["calls"] += 1
        try:
            tet = np.asarray(tet_coords, dtype=float)
            raw_ids = np.asarray(vertex_ids)
            numeric_ids = np.asarray(raw_ids, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return self._reject("invalid_candidate_input")
        if (tet.shape != (4, 3) or not np.isfinite(tet).all() or numeric_ids.shape != (4,)
                or not np.isfinite(numeric_ids).all()
                or not np.equal(numeric_ids, np.floor(numeric_ids)).all()
                or np.any(numeric_ids < -1) or np.any(numeric_ids >= len(self.points))):
            return self._reject("invalid_candidate_input")
        ids = numeric_ids.astype(np.int64)
        if len(set(map(int, ids))) != 4:
            return self._reject("duplicate_candidate_vertex_id")
        for i, vertex_id in enumerate(ids):
            if vertex_id >= 0:
                if np.linalg.norm(tet[i] - self.points[vertex_id]) > self.tol:
                    return self._reject("vertex_id_coordinate_mismatch", vertex_id=int(vertex_id))
            elif np.any(np.linalg.norm(self.points - tet[i], axis=1) <= self.tol):
                return self._reject("new_vertex_duplicates_existing_vertex")

        ab, ac, ad = tet[1:] - tet[0]
        determinant = float(np.dot(ab, np.cross(ac, ad)))
        scale = float(np.max(np.ptp(tet, axis=0)))
        determinant_tol = max(self.tol * scale**2, 128 * np.finfo(float).eps * scale**3)
        if not np.isfinite(determinant) or determinant <= determinant_tol:
            return self._reject("nonpositive_or_degenerate_tet")
        if not self.contains(tet.mean(axis=0)):
            return self._reject("centroid_outside_remaining_domain")

        tetra_faces = tet[np.asarray(_TET_FACES)]
        normals = np.cross(tetra_faces[:, 1] - tetra_faces[:, 0],
                           tetra_faces[:, 2] - tetra_faces[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0.0):
            return self._reject("nonpositive_or_degenerate_tet")
        normals /= lengths[:, None]
        anchors = tetra_faces[:, 0]
        low, high = tet.min(axis=0), tet.max(axis=0)
        nearby = np.flatnonzero(np.all(
            (high + self.tol >= self.lows) & (self.highs + self.tol >= low), axis=1,
        ))
        for face_index in nearby:
            face_index = int(face_index)
            self.stats["pair_checks"] += 1
            allowed, reason = self._pair_is_conforming(tet, ids, normals, anchors, face_index)
            if not allowed:
                return self._reject(reason, face_index=face_index)
        return True
