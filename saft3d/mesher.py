"""SAFT-style batching with a visibility-kernel invariant.

This is a restricted, new implementation, not a general 3D port of SAFT.
Each accepted tetrahedron is grown from an active triangle. During growth,
the remaining cavity stays star-shaped about a fixed certified interior point.
A final phase cones the residual front to that point, terminating the cavity.
No Delaunay tetrahedralizer or precomputed volume tessellation is used.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import numpy as np
from scipy.optimize import linprog
from .backend import create_backend
from .geometry import tet_faces
from .surface_validation import check_embedded_surface


class UnsupportedDomain(ValueError):
    pass


def face_key(face):
    return tuple(sorted(map(int,face)))


def orientation(face):
    a,b,c=map(int,face)
    return 1 if ((a<b)+(a<c)+(b<c)) % 2 == 1 else -1


@dataclass
class MesherConfig:
    backend: str = "cpu"
    batch_size: int = 64
    growth_rounds: int = 2
    size: float = 0.6  # in original coordinate units
    min_quality: float = 1e-5
    max_iterations: int = 10000
    max_tets: int = 10000
    tolerance: float = 1e-10


def check_surface(v, f, tol, *, allow_multiple_components=False):
    if v.ndim!=2 or v.shape[1]!=3 or not np.isfinite(v).all():
        raise ValueError("vertices must be a finite Nx3 array")
    if f.ndim!=2 or f.shape[1]!=3 or len(f)<4:
        raise ValueError("boundary must have at least four triangles")
    if f.min()<0 or f.max()>=len(v): raise ValueError("boundary index out of range")
    if len(set(map(face_key,f))) != len(f): raise ValueError("duplicate boundary triangle")
    edges={}
    for tri in f:
        a,b,c=v[tri]
        if np.linalg.norm(np.cross(b-a,c-a)) < tol:
            raise ValueError("degenerate boundary triangle")
        for x,y in zip(tri,np.roll(tri,-1)):
            k=tuple(sorted((int(x),int(y))))
            edges.setdefault(k,[]).append(1 if x<y else -1)
    if any(len(s)!=2 or sum(s)!=0 for s in edges.values()):
        raise ValueError("surface must be closed, consistently oriented and edge-manifold")
    volume=sum(np.dot(v[a],np.cross(v[b],v[c])) for a,b,c in f)/6
    if volume <= tol: raise ValueError("surface must have positive outward signed volume")
    # Single connected component: the prototype has one visibility kernel.
    incident={}
    for i,tri in enumerate(f):
        for x in tri: incident.setdefault(int(x),[]).append(i)
    pending=[0]; seen=set()
    while pending:
        i=pending.pop()
        if i in seen: continue
        seen.add(i)
        for x in f[i]: pending.extend(j for j in incident[int(x)] if j not in seen)
    if len(seen)!=len(f) and not allow_multiple_components:
        raise UnsupportedDomain("multiple boundary components are not supported")
    return volume


def visibility_kernel(v, f, tol):
    triangles=v[f]
    normals=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    # Chebyshev center of all inward half-spaces. LP is NOT a tetrahedralizer.
    result=linprog([0,0,0,-1],A_ub=np.column_stack((normals,np.ones(len(f)))),
                   b_ub=np.einsum('ij,ij->i',normals,triangles[:,0]),
                   bounds=[(None,None)]*3+[(0,None)],method="highs")
    if not result.success or result.x[3] <= 100*tol:
        raise UnsupportedDomain("no positive-radius visibility kernel; arbitrary non-star-shaped domains are unsupported")
    return result.x[:3],float(result.x[3])


def grow_mesh(vertices, boundary_faces, config=None):
    cfg=config or MesherConfig()
    if not np.isfinite(cfg.tolerance) or cfg.tolerance<=0:
        raise ValueError("tolerance must be finite and positive")
    if cfg.max_iterations<1 or cfg.max_tets<1:
        raise ValueError("iteration and tetrahedron limits must be positive")
    if not 1<=cfg.batch_size<=512: raise ValueError("batch_size must be in [1,512]")
    if not 0<=cfg.growth_rounds<=20: raise ValueError("growth_rounds must be in [0,20]")
    if cfg.size<=0 or not np.isfinite(cfg.size): raise ValueError("size must be finite and positive")
    if not 0<=cfg.min_quality<1: raise ValueError("min_quality must be in [0,1)")
    start=time.perf_counter()
    original=np.asarray(vertices,dtype=np.float64)
    raw_faces=np.asarray(boundary_faces)
    if raw_faces.dtype.kind not in 'iu':
        raise ValueError("boundary faces must contain integer indices")
    faces=raw_faces.astype(np.int64)
    if original.ndim!=2 or original.shape[1]!=3 or len(original)<4 or not np.isfinite(original).all():
        raise ValueError("vertices must be Nx3, N>=4")
    offset=original.min(axis=0); scale=float(np.ptp(original,axis=0).max())
    if not np.isfinite(scale) or scale<=0: raise ValueError("invalid domain extent")
    normalized=(original-offset)/scale
    domain_volume=check_surface(normalized,faces,cfg.tolerance)
    check_embedded_surface(normalized,faces,cfg.tolerance)
    anchor,radius=visibility_kernel(normalized,faces,cfg.tolerance)
    points=list(normalized.copy()); anchor_id=len(points); points.append(anchor.copy())
    front={face_key(f):tuple(map(int,f)) for f in faces}
    generation={k:0 for k in front}
    tets=[]; history=[]; backend=create_backend(cfg.backend)
    target=cfg.size/scale
    total_candidates=0; growth_count=0; closure_count=0
    # Depth bounds each radial subdivision, so closure is finite in exact arithmetic.
    for iteration in range(cfg.max_iterations):
        if not front: break
        p=np.asarray(points)
        items=[]
        for key,tri in front.items():
            a,b,c=p[list(tri)]
            n=np.cross(b-a,c-a); n/=np.linalg.norm(n)
            depth=-float(np.dot(n,anchor-a))
            if depth>20*cfg.tolerance: items.append((generation[key],key,tri,depth,n))
        if not items:
            raise RuntimeError(f"stalled: {len(front)} residual faces, none can grow toward the kernel")
        # Lower generations first: avoid refining a single patch indefinitely.
        items.sort(key=lambda x:(x[0],x[1]))
        items=items[:cfg.batch_size]
        base_triangles=p[np.asarray([item[2] for item in items])]
        close_flags=np.asarray([gen>=cfg.growth_rounds or depth<.35*target
                                for gen,key,tri,depth,n in items])
        batch,valid=backend.propose(base_triangles,anchor,target,close_flags,cfg.tolerance)
        rows,slots=np.nonzero(valid)
        proposed=batch[rows,slots]
        meta=[]
        for row,slot in zip(rows,slots):
            gen,key,tri,depth,n=items[row]
            meta.append((key,tri,gen,anchor_id if slot==5 else -1,
                         "closure" if slot==5 else "growth"))
        if not len(proposed): raise RuntimeError("no geometric proposals")
        volumes,quality=backend.evaluate(proposed)
        total_candidates+=len(proposed)
        # SAFT single-face winner; shape and target-size score are local heuristics.
        winners={}
        for i,(key,tri,gen,did,phase) in enumerate(meta):
            if volumes[i]<=cfg.tolerance**2 or quality[i]<cfg.min_quality: continue
            edge=np.linalg.norm(proposed[i,0]-proposed[i,3])
            score=float(quality[i])*min(edge/target,target/max(edge,1e-30))**0.25
            if key not in winners or score>winners[key][0]: winners[key]=(score,i)
        ids=[value[1] for value in winners.values()]
        if not ids: raise RuntimeError("all candidates failed the configured quality threshold")
        selected_tets=proposed[ids]
        collision=backend.conflicts(selected_tets,padding=0.0)
        if tets:
            invalid=backend.overlaps(selected_tets,p[np.asarray(tets)],tol=cfg.tolerance)
        else: invalid=np.zeros(len(ids),dtype=bool)
        chosen=[]
        for j,i in enumerate(ids):
            if invalid[j] or any(collision[j,k] for k in chosen): continue
            chosen.append(j)
        if not chosen: raise RuntimeError("no progress: all candidates overlap accepted elements")
        accepted=0
        for j in chosen:
            i=ids[j]; key,tri,gen,did,phase=meta[i]
            if key not in front: continue
            if did<0:
                did=len(points); points.append(proposed[i,3].copy())
            indices=(tri[0],tri[2],tri[1],did)
            updates=[]
            for face in tet_faces(indices):
                fk=face_key(face)
                if fk in front:
                    if orientation(front[fk])!=orientation(face):
                        raise RuntimeError("same-side front collision / inconsistent orientation")
                    updates.append((fk,None))
                else:
                    updates.append((fk,(face[0],face[2],face[1])))
            for fk,value in updates:
                if value is None:
                    del front[fk]; generation.pop(fk,None)
                else:
                    front[fk]=value; generation[fk]=gen+1
            tets.append(indices); accepted+=1
            growth_count+=phase=="growth"; closure_count+=phase=="closure"
            if len(tets)>cfg.max_tets: raise RuntimeError("max_tets exceeded")
        history.append({"iteration":iteration,"front_faces":len(front),
                        "proposals":len(proposed),"winners":len(ids),"accepted":accepted})
        if not accepted: raise RuntimeError("no topological progress")
    if front: raise RuntimeError(f"max_iterations exceeded with {len(front)} faces remaining")
    mesh_vertices=np.asarray(points)*scale+offset
    mesh_vertices[:len(original)]=original
    return mesh_vertices,np.asarray(tets,dtype=np.int64),{
        "algorithm":"SAFT-inspired star-kernel advancing front (restricted 3D prototype)",
        "config":asdict(cfg),"backend":backend.name,"device":backend.device_info,
        "backend_stats":backend.stats,
        "kernel_point":(anchor*scale+offset).tolist(),"kernel_radius":radius*scale,
        "iterations":len(history),"proposals":total_candidates,
        "growth_tets":growth_count,"closure_tets":closure_count,
        "remaining_front_faces":0,"initial_volume":domain_volume*scale**3,
        "meshing_seconds":time.perf_counter()-start,"history":history,
        "limitations":["requires a valid closed oriented star-shaped surface",
                       "float64 tolerance-based predicates, not exact arithmetic",
                       "target size is heuristic, no global mesh-size or quality guarantee",
                       "CPU topology with optional CUDA scoring, conflicts and overlap checks"]}
