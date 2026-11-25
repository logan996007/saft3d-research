"""Numerical local legality checks for general advancing-front tetrahedra.

No visibility kernel or tetrahedralization library is used.  The current front
must be a valid, closed, outward-oriented embedded surface; cavity shells point
into the excluded holes.  The caller remains responsible for transactional
front topology updates and conflicts between candidates in the same batch.

Triangle/tetrahedron intersection is checked independently with barycentric
linear programs.  The predicates are tolerance based, not exact arithmetic.
"""

from itertools import combinations
from collections import OrderedDict
from math import fsum

import numpy as np
from scipy.optimize import linprog


_TET_FACES = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
_FACE_ORDER = np.asarray(_TET_FACES)
_TET_EDGES = np.asarray(list(combinations(range(4),2)))
_RAY_PAIRS = np.asarray(list(combinations(range(7),2)))
_LP_OPTIONS = {
    "primal_feasibility_tolerance": 1.0e-10,
    "dual_feasibility_tolerance": 1.0e-10,
    "ipm_optimality_tolerance": 1.0e-12,
}


class FrontContactCache:
    """Per-run cache of tet/triangle checks for append-only vertex coordinates.

    Full tet coordinates and IDs key the queries. An oriented face gets a
    persistent token only on the same growing mesh; unchanged vertex IDs
    always refer to the same coordinates. Cache entries never cover winding,
    topology changes or newly introduced faces.
    """
    def __init__(self, max_entries=1024):
        self.tokens={}
        self.entries=OrderedDict()
        self.limit=max_entries

    def face_tokens(self, faces):
        tokens=self.tokens
        return [tokens.setdefault(tuple(map(int,f)), len(tokens)) for f in faces]

    def save(self,key,seen,bad=None):
        if len(seen)>512:return
        entries=self.entries
        entries[key]=(seen,bad)
        entries.move_to_end(key)
        if len(entries)>self.limit:entries.popitem(last=False)


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
        if np.any((self.faces[:,0]==self.faces[:,1]) | (self.faces[:,1]==self.faces[:,2])
                  | (self.faces[:,2]==self.faces[:,0])):
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
        self.block_order=None
        if len(self.faces)>=192:
            # Spatially coherent buckets contain complete triangle AABBs,
            # including long faces whose centroid lies far from the query.
            centres=.5*(self.lows+self.highs)
            axis=np.ptp(centres,axis=0).argmax()
            self.block_order=np.argsort(centres[:,axis],kind='stable')
            offsets=np.arange(0,len(self.faces),16)
            self.block_lows=np.minimum.reduceat(self.lows[self.block_order],offsets,axis=0)
            self.block_highs=np.maximum.reduceat(self.highs[self.block_order],offsets,axis=0)
        self.contact_cache=None
        self.stats = {
            "calls": 0, "contains_calls": 0, "pair_checks": 0, "lp_calls": 0,
            "rejections": 0, "reasons": {}, "last_rejection": None,
            "certified_pairs":0, "reused_pairs":0,
        }

    def _reject(self, reason, **details):
        self.stats["rejections"] += 1
        self.stats["reasons"][reason] = self.stats["reasons"].get(reason, 0) + 1
        self.stats["last_rejection"] = {"reason": reason, **details}
        return False

    def _nearby_faces(self,low,high):
        """AABB broad phase retaining exactly the exhaustive scan's face order."""
        if self.block_order is None:
            return np.flatnonzero(np.all((high+self.tol>=self.lows)
                                  & (self.highs+self.tol>=low),axis=1))
        keep=np.flatnonzero(np.all((high+self.tol>=self.block_lows)
                           & (self.block_highs+self.tol>=low),axis=1))
        slots=(16*keep[:,None]+np.arange(16)).ravel()
        indices=self.block_order[slots[slots<len(self.faces)]]
        inside=np.all((high+self.tol>=self.lows[indices])
                & (self.highs[indices]+self.tol>=low),axis=1)
        return np.sort(indices[inside])

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

    def _batch_certificates(self,tet,ids,normals,anchors,indices):
        """Resolve well-separated AABB hits; leave all uncertain geometry to LP.

        The first stage applies precisely the scalar face-plane tests to a
        whole array. Additional strict certificates use larger margins than
        the LP feasibility tolerance and require bit-identical shared points.
        """
        tri=self.triangles[indices]
        matches=self.faces[indices,:,None]==ids
        s_tri=matches.any(axis=2);s_tet=matches.any(axis=1)
        nshared=s_tri.sum(axis=1)
        tol=max(self.tol,64*np.finfo(float).eps)
        face_side=np.einsum('nkj,nj->nk',tet[None]-tri[:,:1],self.normals[indices])
        full=nshared==3
        wrong=full & (np.max(np.where(~s_tet,face_side,-np.inf),axis=1)>=-self.tol)
        good=full & ~wrong
        outside=np.einsum('nfkj,fj->nfk',tri[:,None,:,:]-anchors[None,:,None,:],normals)
        good |= (outside>tol).all(axis=2).any(axis=1)
        good |= np.all(np.where(s_tri[:,None,:],abs(outside)<=tol,outside>tol),axis=2).any(axis=1)
        good |= np.all(np.where(s_tet,abs(face_side)<=tol,face_side>tol),axis=1)
        good |= np.all(np.where(s_tet,abs(face_side)<=tol,face_side< -tol),axis=1)
        # Unequal coordinates with a shared index are within the caller's
        # tolerance. Only use the original narrow-phase predicates for them.
        exact=(~matches | np.all(tri[:,:,None,:]==tet[None,None,:,:],axis=3)).all(axis=(1,2))
        scale=np.maximum(np.ptp(tet,axis=0).max(),np.max(np.maximum(tet.max(axis=0),self.highs[indices])
                      -np.minimum(tet.min(axis=0),self.lows[indices]),axis=1))
        # An explicit positive clearance is deliberately larger than HiGHS'
        # normalized tolerances. Near-coplanar cases go to the scalar fallback.
        margin=np.maximum(16*self.tol,1.e-8*scale)
        for shared_count in (0,1,2):
            selected=np.flatnonzero(~good & (nshared==shared_count) & exact)
            if not len(selected):continue
            triangles=tri[selected]
            if shared_count==0:
                # Tri/tet SAT's edge-cross-edge directions can certify
                # disjointness, where neither primitive's own face does.
                e=triangles[:,[1,2,0]]-triangles
                f=tet[_TET_EDGES[:,1]]-tet[_TET_EDGES[:,0]]
                axes=np.cross(e[:,:,None,:],f[None,None,:,:]).reshape(len(e),18,3)
                length=np.linalg.norm(axes,axis=2)
                a=np.einsum('nkj,naj->nka',triangles-tet[0],axes)
                b=np.einsum('kj,naj->nka',tet-tet[0],axes)
                gaps=np.maximum(a.min(axis=1)-b.max(axis=1),b.min(axis=1)-a.max(axis=1))
                good[selected]|=np.any(gaps>length*margin[selected,None],axis=1)
                continue
            tri_shared=s_tri[selected]; tet_shared=s_tet[selected]
            rows=np.arange(len(selected))
            common=triangles[rows,tri_shared.argmax(axis=1)]
            dirs=np.concatenate((triangles-common[:,None],common[:,None]-tet),axis=1)
            shared=np.column_stack((tri_shared,tet_shared))
            axis=None
            if shared_count==2:
                second=triangles[rows,2-tri_shared[:,::-1].argmax(axis=1)]
                axis=second-common
                axis/=np.linalg.norm(axis,axis=1)[:,None]
            normalized=dirs.copy()
            if axis is not None:
                normalized-=np.einsum('nkj,nj->nk',normalized,axis)[:,:,None]*axis[:,None]
            lengths=np.linalg.norm(normalized,axis=2)
            normalized/=np.where(lengths>0,lengths,1.)[:,:,None]
            normalized[shared]=0.

            def strict(n):
                # Reproject after summing cone rays: cancellation can amplify
                # roundoff along a very short shared edge. The separation
                # plane must contain the *entire* indexed segment.
                if axis is not None:
                    n=n-np.einsum('nj,nj->n',n,axis)[:,None]*axis
                length=np.linalg.norm(n,axis=1)
                n=n/np.where(length>0,length,1.)[:,None]
                d=np.einsum('nkj,nj->nk',dirs,n)
                ok=np.all(np.where(shared,abs(d)<=64*np.finfo(float).eps*scale[selected,None],
                                   d>margin[selected,None]),axis=1)
                if axis is not None:
                    ok &= abs(np.einsum('nj,nj->n',n,axis))<=64*np.finfo(float).eps
                return ok

            passed=strict(normalized.sum(axis=1))
            if not passed.all():
                if axis is None:
                    rays=np.cross(normalized[:,_RAY_PAIRS[:,0]],normalized[:,_RAY_PAIRS[:,1]])
                else:
                    rays=np.cross(axis[:,None,:],normalized)
                # Extreme rays are candidate separating directions only. The
                # final, strict vertex-side checks are the actual certificate.
                lengths=np.linalg.norm(rays,axis=2)
                rays/=np.where(lengths>0,lengths,1.)[:,:,None]
                products=normalized@rays.swapaxes(1,2)
                signs=(products.min(axis=1)>=-1e-13).astype(float)-(products.max(axis=1)<=1e-13)
                passed |= strict(np.einsum('nr,nrj->nj',signs,rays))
            good[selected]|=passed
        # Three shared vertices require the opposite tet vertex to be inward.
        good[wrong]=False
        return good

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

        tetra_faces = tet[_FACE_ORDER]
        normals = np.cross(tetra_faces[:, 1] - tetra_faces[:, 0],
                           tetra_faces[:, 2] - tetra_faces[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        if np.any(lengths <= 0.0):
            return self._reject("nonpositive_or_degenerate_tet")
        normals /= lengths[:, None]
        anchors = tetra_faces[:, 0]
        low, high = tet.min(axis=0), tet.max(axis=0)
        nearby = self._nearby_faces(low,high)
        cache=self.contact_cache
        if cache is not None:
            cache_key=(ids.tobytes(),tet.tobytes())
            prior=cache.entries.get(cache_key)
            tags=self.face_tokens
            all_nearby=nearby
            if prior is not None:
                seen,bad=prior
                if bad is not None:
                    token,reason=bad
                    found=next((j for j in nearby if tags[j]==token),None)
                    if found is not None:return self._reject(reason,face_index=int(found))
                nearby=np.fromiter((j for j in nearby if tags[j] not in seen),dtype=np.intp)
                self.stats['reused_pairs']+=len(all_nearby)-len(nearby)
                if not len(nearby):return True
        certified=self._batch_certificates(tet,ids,normals,anchors,nearby)
        self.stats["pair_checks"]+=int(certified.sum())
        self.stats["certified_pairs"]+=int(certified.sum())
        for face_index in nearby[~certified]:
            face_index = int(face_index)
            self.stats["pair_checks"] += 1
            allowed, reason = self._pair_is_conforming(tet, ids, normals, anchors, face_index)
            if not allowed:
                if cache is not None and reason!='intersection_lp_failure':
                    cache.save(cache_key,frozenset(),(tags[face_index],reason))
                return self._reject(reason, face_index=face_index)
        if cache is not None:
            cache.save(cache_key,frozenset(tags[j] for j in all_nearby))
        return True
