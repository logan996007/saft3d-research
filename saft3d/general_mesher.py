"""Local advancing-front tetrahedral generation on oriented triangle surfaces.

The front advances using normal-direction proposals and existing vertices.
On a stall, the configured completion mode either reports the residual front
or submits its frozen triangle boundary to the CPU TetGen worker.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import defaultdict,deque
import time
import numpy as np
from .backend import create_backend
from .geometry import tet_faces
from .mesher import check_surface,face_key,orientation
from .surface_validation import check_embedded_surface
from .front_geometry import FrontGeometry,FrontContactCache


@dataclass
class AdvancingFrontConfig:
    backend: str = "cpu"
    size: float = .6
    batch_size: int = 32
    candidate_vertices: int = 24
    min_quality: float = 1e-4
    new_vertex_min_quality: float = .12
    max_new_vertices: int = 256
    max_iterations: int = 2000
    max_tets: int = 10000
    tolerance: float = 1e-10
    completion: str = "none"
    completion_timeout: float = 60.0
    max_completion_rollback: int = 32


class StalledFront(RuntimeError):
    """Diagnostic exception retaining an honest partial mesh and residual front."""
    def __init__(self,message,vertices,tetrahedra,front,report):
        super().__init__(message)
        self.vertices=vertices
        self.tetrahedra=tetrahedra
        self.front=front
        self.report=report


def _front_delta(front, indices):
    """Compute four oriented face toggles without copying the complete front.

    The caller maintains an initially checked closed chain. Subtracting a
    tetrahedron boundary preserves zero signed incidence at every edge
    (the boundary of a boundary is zero), including edges at shell mergers.
    """
    delta=[]
    for f in tet_faces(indices):
        key=face_key(f)
        existing=front.get(key)
        if existing is None:
            delta.append((key,(f[0],f[2],f[1])))
        elif orientation(f)==orientation(existing):
            delta.append((key,None))
        else:
            return None
    return delta


def _updated_front(front,indices):
    """Return a consistently oriented residual boundary chain, or None.

    Intermediate cavities may meet at an indexed edge/vertex while fronts merge.
    Requiring every intermediate link to be a single cycle would forbid joining
    an outer shell to a cavity shell. Geometrical contact is checked separately.
    """
    result=front.copy()
    for f in tet_faces(indices):
        key=face_key(f)
        if key in result:
            if orientation(f)!=orientation(result[key]): return None
            del result[key]
        else: result[key]=(f[0],f[2],f[1])
    if not result: return result
    edges=defaultdict(list)
    for a,b,c in result.values():
        for x,y in ((a,b),(b,c),(c,a)):
            edges[tuple(sorted((x,y)))].append(1 if x<y else -1)
    if any(sum(v)!=0 for v in edges.values()): return None
    return result


def grow_general_mesh(vertices,boundary_faces,config=None,*,size_field=None):
    """Use physical lengths returned by an optional size_field(Nx3 face centres)."""
    cfg=config or AdvancingFrontConfig()
    start=time.perf_counter()
    if not np.isfinite(cfg.size) or cfg.size<=0: raise ValueError("size must be positive and finite")
    if not np.isfinite(cfg.tolerance) or cfg.tolerance<=0: raise ValueError("tolerance must be positive and finite")
    if not 1<=cfg.batch_size<=512: raise ValueError("batch_size must be in [1,512]")
    if not 1<=cfg.candidate_vertices<=2048: raise ValueError("candidate_vertices must be in [1,2048]")
    if not 0<=cfg.min_quality<=cfg.new_vertex_min_quality<1: raise ValueError("invalid quality thresholds")
    if cfg.max_iterations<1 or cfg.max_tets<1 or cfg.max_new_vertices<0: raise ValueError("invalid resource limits")
    if cfg.completion not in ('none','tetgen'): raise ValueError("completion must be none or tetgen")
    if cfg.max_completion_rollback<0: raise ValueError('max_completion_rollback must be nonnegative')
    if not np.isfinite(cfg.completion_timeout) or cfg.completion_timeout<=0:
        raise ValueError('completion_timeout must be positive and finite')
    raw=np.asarray(vertices,dtype=float); f=np.asarray(boundary_faces)
    if raw.ndim!=2 or raw.shape[1]!=3 or len(raw)<4 or not np.isfinite(raw).all():
        raise ValueError("vertices must be a finite Nx3 array")
    if f.dtype.kind not in 'iu': raise ValueError("faces must contain integer indices")
    offset=raw.min(axis=0); scale=float(np.ptp(raw,axis=0).max())
    if scale<=0: raise ValueError("zero domain extent")
    v=(raw-offset)/scale
    volume=check_surface(v,f,cfg.tolerance,allow_multiple_components=True)
    check_embedded_surface(v,f,cfg.tolerance)
    front={face_key(face):tuple(map(int,face)) for face in f}
    face_age={key:0 for key in front}
    points=list(v.copy()); tets=[]; history=[]
    backend=create_backend(cfg.backend)
    contact_cache=FrontContactCache()
    observed_min, observed_max = float('inf'), 0.
    new_points=0; candidate_count=0; cursor=0; stalled_passes=0; exhaustive=False
    legal_stats=defaultdict(int)
    stop_reason='iteration_limit'
    for iteration in range(cfg.max_iterations):
        if not front: break
        p=np.asarray(points)
        all_front=np.asarray(list(front.values()),dtype=np.int64)
        geom=FrontGeometry(p,all_front,cfg.tolerance)
        geom.contact_cache=contact_cache
        geom.face_tokens=contact_cache.face_tokens(all_front)
        active_ids=np.unique(all_front)
        xyz=geom.triangles
        lengths=np.linalg.norm(xyz-np.roll(xyz,-1,axis=1),axis=2).max(axis=1)
        length_by_face=dict(zip(front,lengths))
        ordered=sorted(front.items(),key=lambda item:(face_age[item[0]],length_by_face[item[0]],item[0]))
        selected=ordered[cursor:cursor+cfg.batch_size]
        if not selected:
            cursor=0; selected=ordered[:cfg.batch_size]
        base=p[np.asarray([tri for key,tri in selected])]
        if size_field is None:
            physical_sizes = np.full(len(selected), cfg.size)
        else:
            physical_sizes = np.asarray(size_field(base.mean(axis=1)*scale+offset), dtype=float)
            if physical_sizes.shape != (len(selected),) or not np.isfinite(physical_sizes).all() or np.any(physical_sizes <= 0):
                raise ValueError("size_field must return one finite positive physical length per face")
        observed_min = min(observed_min, float(physical_sizes.min()))
        observed_max = max(observed_max, float(physical_sizes.max()))
        targets = physical_sizes / scale
        proposed,valid=backend.propose_local(base,targets,cfg.tolerance)
        blocks=[]; locations=[]; apex_ids=[]
        active_points=p[active_ids]
        for row,(key,tri) in enumerate(selected):
            ideal=proposed[row,0,3]
            distance=np.linalg.norm(active_points-ideal,axis=1)
            nearest=np.argsort(distance,kind='stable')
            normal=np.cross(base[row,1]-base[row,0],base[row,2]-base[row,0])
            normal/=np.linalg.norm(normal)
            offsets=active_points-base[row,0]
            sides=offsets@normal
            # Scalar dot recovers the historical decision on roundoff ties.
            close=np.flatnonzero(abs(sides+cfg.tolerance)<32*np.finfo(float).eps)
            for j in close:sides[j]=np.dot(normal,offsets[j])
            inward=sides< -cfg.tolerance
            for idx in tri:inward &= active_ids!=idx
            existing=active_ids[nearest[inward[nearest]]]
            if not exhaustive:existing=existing[:cfg.candidate_vertices]
            if len(existing):
                block=np.empty((len(existing),4,3))
                block[:,:3]=base[row,[0,2,1]];block[:,3]=p[existing]
                blocks.append(block)
                locations.extend([row]*len(existing));apex_ids.extend(map(int,existing))
            if new_points<cfg.max_new_vertices:
                block=proposed[row,valid[row]]
                if len(block):
                    distances=np.linalg.norm(p[None]-block[:,3,None,:],axis=2)
                    block=block[distances.min(axis=1)>100*cfg.tolerance]
                if len(block):
                    blocks.append(block)
                    locations.extend([row]*len(block));apex_ids.extend([-1]*len(block))
        if blocks:
            candidates=np.concatenate(blocks)
            vols,qualities=backend.evaluate(candidates)
            overlaps_existing=(backend.overlaps(candidates,p[np.asarray(tets)],cfg.tolerance)
                               if tets else np.zeros(len(candidates),bool))
            candidate_count+=len(candidates)
            grouped=[[] for _ in selected]
            required=np.where(np.asarray(apex_ids)<0,cfg.new_vertex_min_quality,cfg.min_quality)
            eligible=np.flatnonzero((vols>1e-12)&(qualities>=required)&~overlaps_existing)
            lengths=np.linalg.norm(candidates[eligible,3,None,:]-candidates[eligible,:3],axis=2).mean(axis=1)
            target=targets[np.asarray(locations)[eligible]]
            ratios=np.minimum(lengths/target,target/np.maximum(lengths,1e-30))
            for i,sizing in zip(eligible,ratios):
                pid=apex_ids[i]
                # Favor reusing vertices enough to merge fronts, but not at any quality.
                score=float(qualities[i])*sizing**.35*(1.08 if pid>=0 else 1.)
                grouped[locations[i]].append((score,int(i)))
            winners=[]
            for row,(_,tri) in enumerate(selected):
                for score,i in sorted(grouped[row],reverse=True):
                    pid=apex_ids[i]
                    idx=(tri[0],tri[2],tri[1],pid if pid>=0 else len(points))
                    if _front_delta(front,idx) is None: continue
                    if geom.valid_candidate(candidates[i],(tri[0],tri[2],tri[1],pid)):
                        winners.append(i); break
            if winners:
                tested=candidates[winners]
                conflict=backend.conflicts(tested)
                overlap=overlaps_existing[winners]
                chosen=[]
                for j,i in enumerate(winners):
                    if overlap[j] or any(conflict[j,k] for k in chosen): continue
                    chosen.append(j)
                accepted=0
                for j in chosen:
                    i=winners[j]; key,tri=selected[locations[i]];pid=apex_ids[i]
                    if key not in front: continue
                    if pid<0 and new_points>=cfg.max_new_vertices: continue
                    if len(tets)>=cfg.max_tets:
                        stop_reason='tetrahedron_limit'; break
                    idx=(tri[0],tri[2],tri[1],pid if pid>=0 else len(points))
                    delta=_front_delta(front,idx)
                    if delta is None: continue
                    if pid<0:
                        points.append(candidates[i,3].copy()); new_points+=1
                    tets.append(idx); accepted+=1
                    for changed,face in delta:
                        if face is None:
                            del front[changed];del face_age[changed]
                        else:
                            front[changed]=face;face_age[changed]=iteration+1
                if accepted:
                    history.append({"iteration":iteration,"front_faces":len(front),"proposals":len(candidates),
                                    "winners":len(winners),"accepted":accepted,"new_vertices":new_points})
                    cursor=0; stalled_passes=0; exhaustive=False
                    for key,value in getattr(geom,'stats',{}).items():
                        if isinstance(value,(int,float)): legal_stats[key]+=value
                    if len(tets)>=cfg.max_tets:
                        stop_reason='tetrahedron_limit'; break
                    continue
        for key,value in getattr(geom,'stats',{}).items():
            if isinstance(value,(int,float)): legal_stats[key]+=value
        cursor+=cfg.batch_size
        if cursor>=len(ordered):
            stalled_passes+=1; cursor=0
            if exhaustive or len(active_ids)<=cfg.candidate_vertices+3:
                stop_reason='no_legal_candidate'; break
            exhaustive=True
    output=np.asarray(points)*scale+offset
    output[:len(raw)]=raw  # preserve prescribed coordinates bit-for-bit
    tet_indices=np.asarray(tets,dtype=np.int64).reshape(-1,4)
    report={"algorithm":"local 3D SAFT-inspired advancing front (no common kernel)",
            "config":asdict(cfg),"backend":backend.name,"device":backend.device_info,
            "backend_stats":backend.stats,"initial_volume":volume*scale**3,
            "face_size_field":getattr(size_field, "__name__", "constant" if size_field is None else type(size_field).__name__),
            "sampled_physical_sizes":[observed_min,observed_max],
            "iterations":len(history),"proposals":candidate_count,"new_vertices":new_points,
            "growth_tets":len(tets),"closure_tets":0,"remaining_front_faces":len(front),
            "aft_stop_reason":stop_reason if front else 'front_closed',
            "meshing_seconds":time.perf_counter()-start,"history":history,"legality_stats":dict(legal_stats),
            "limitations":["heuristic local advancement: may stall on a residual cavity",
                           "float64 geometry with independent LP checks, not exact predicates",
                           "size is a local heuristic and boundary is fixed"]}
    if front:
        if cfg.completion=='tetgen':
            from .cavity_completion import complete_cavity
            limit=min(cfg.max_completion_rollback,max(0,len(tets)-1))
            rollback_steps=sorted({0,limit,*[2**k for k in range(20) if 2**k<=limit]})
            pending=deque(tuple(range(len(tets)-count,len(tets))) for count in rollback_steps)
            attempted=set()
            owners=defaultdict(set)
            for tet_id,tet in enumerate(tet_indices):
                for face in tet_faces(tet): owners[face_key(face)].add(tet_id)
            deadline=time.perf_counter()+cfg.completion_timeout
            report['completion_attempts']=[]
            report['provisional_aft_tets']=len(tets)
            while pending:
                removed=pending.popleft()
                if removed in attempted: continue
                attempted.add(removed); rollback=len(removed)
                retained=np.delete(tet_indices,removed,axis=0) if rollback else tet_indices
                retry_front=front
                if rollback:
                    retry_front={face_key(face):tuple(map(int,face)) for face in f}
                    for tet in retained:
                        change=_front_delta(retry_front,tuple(map(int,tet)))
                        if change is None:raise RuntimeError('cannot reconstruct a valid rollback boundary chain')
                        for changed, face in change:
                            if face is None:del retry_front[changed]
                            else:retry_front[changed]=face
                attempt={'rolled_back_tets':rollback,'retained_aft_tets':len(retained),
                         'rolled_back_tet_indices':list(removed),'front_faces':len(retry_front)}
                try:
                    remaining=deadline-time.perf_counter()
                    if remaining<=0: raise RuntimeError('total cavity completion timeout exceeded')
                    completed_points,completed_tets,completion_report=complete_cavity(
                        output,np.asarray(list(retry_front.values()),dtype=np.int64),timeout=remaining,
                        max_steiner=max(0,cfg.max_tets-len(retained)))
                    if len(retained)+len(completed_tets)>cfg.max_tets:
                        raise RuntimeError("cavity completion exceeded max_tets")
                    if not np.array_equal(completed_points[:len(output)],output):
                        raise RuntimeError("cavity completion altered retained AFT vertex coordinates")
                    attempt['status']='completed';report['completion_attempts'].append(attempt)
                    report.pop('completion_error',None)
                    report['cpu_completion']=completion_report
                    report['rolled_back_aft_tets']=rollback
                    report['rolled_back_aft_indices']=list(removed)
                    report['growth_tets']=len(retained)
                    report['closure_tets']=len(completed_tets)
                    report['remaining_front_faces']=0
                    report['algorithm']='3D SAFT-inspired local AFT with explicit CPU constrained cavity completion'
                    report['meshing_seconds']=time.perf_counter()-start
                    old_coords=output[retained]
                    aft_volume=float(np.einsum('ij,ij->i',old_coords[:,1]-old_coords[:,0],
                                  np.cross(old_coords[:,2]-old_coords[:,0],old_coords[:,3]-old_coords[:,0])).sum()/6)
                    report['aft_volume_fraction']=aft_volume/(volume*scale**3)
                    report['completion_volume_fraction']=1-report['aft_volume_fraction']
                    report['limitations'][0]='local AFT may stall; explicit bounded rollback expands CPU-completed residual cavities'
                    return completed_points,np.vstack((retained,completed_tets)),report
                except Exception as exc:
                    attempt['status']='failed';attempt['error']=f'{type(exc).__name__}: {exc}'
                    diagnostics=getattr(exc,'diagnostics',{})
                    if diagnostics:
                        attempt['diagnostics']=diagnostics
                        implicated=set(removed)
                        for face in diagnostics.get('missing_global_faces',[]):
                            implicated.update(owners[face_key(face)])
                        targeted=tuple(sorted(implicated))
                        if len(removed)<len(targeted)<=limit and targeted not in attempted:
                            pending.appendleft(targeted)
                    report['completion_attempts'].append(attempt)
                    report['completion_error']=attempt['error']
                    if time.perf_counter()>=deadline: break
        raise StalledFront(f"local AFT stalled with {len(front)} front faces after {len(tets)} tetrahedra",
                           output,tet_indices,np.asarray(list(front.values()),dtype=np.int64),report)
    return output,tet_indices,report
