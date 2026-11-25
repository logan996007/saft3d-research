"""Batch local six-vertex advancing fronts on straight extrusions of a planar cap.

The two cap triangles and six triangulated walls are the only prescribed faces.
All fifteen four-vertex subsets are considered, including degenerate ones.
The partial mesh is constructed by oriented front updates. There is no table
of three chosen tetrahedra indexed by a prism's diagonal pattern.
"""
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
from .backend import prism_backend

_TRIPLES = list(combinations(range(6),3))
_FACES = np.array(((1,2,3),(0,3,2),(0,1,3),(0,2,1)))
_SUPPORT = ((0,1,2),(3,4,5),(0,1,4,3),(1,2,5,4),(2,0,3,5))
_EXTERIOR = sum(1<<i for i,f in enumerate(_TRIPLES) if any(set(f)<=set(s) for s in _SUPPORT))


def _bit(face):
    a,b,c=map(int,face)
    parity=(int(a>b)+int(a>c)+int(b>c))%2
    return (1<<_TRIPLES.index(tuple(sorted((a,b,c)))),parity)


def _mask(faces):
    result=[0,0]
    for f in faces:
        bit,side=_bit(f);result[side]|=bit
    return result


def _catalogue():
    """Enumerate every candidate, oriented in a right-handed reference prism."""
    ref=np.array(((0,0,0),(1,0,0),(0,1,0),(0,0,1),(1,0,1),(0,1,1)))
    local=[];masks=[]
    for item in combinations(range(6),4):
        ids=np.array(item)
        pts=ref[ids]
        if np.linalg.det(pts[1:]-pts[0])<0:ids[[0,1]]=ids[[1,0]]
        local.append(ids);masks.append(_mask(ids[_FACES]))
    return np.array(local,dtype=np.int32),np.array(masks,dtype=np.uint32)


_CANDIDATES,_ORIENTED=_catalogue()


def _check_planar_cap(xy,cap):
    """Prove projected triangles are disjoint or meet only on indexed simplices.

    Every separating axis of two convex triangles is normal to one of their
    six edges. For a common vertex, a separating line can also contain other
    vertices; their projected contact interval must end at that shared vertex.
    """
    tri=xy[cap]; low=tri.min(axis=1); high=tri.max(axis=1)
    span=float(np.ptp(xy,axis=0).max());tol=128*np.finfo(float).eps*span
    order=np.argsort(low[:,0],kind='stable');x=low[order,0]

    def certified(a,b,ia,ib):
        shared=(ia[:,:,None]==ib[:,None,:]).any(axis=2)
        edge=np.roll(a,-1,axis=1)-a
        lengths=np.linalg.norm(edge,axis=2)
        unit=edge/lengths[:,:,None]
        # Coordinates of B vertices in each A edge's oriented local frame.
        rel=b[:,None,:,:]-a[:,:,None,:]
        side=unit[:,:,0,None]*rel[:,:,:,1]-unit[:,:,1,None]*rel[:,:,:,0]
        strict=(side.max(axis=2)<-tol).any(axis=1)
        pos=np.einsum('bevc,bec->bev',rel,unit)
        on=side>=-tol
        at_start=shared & np.all(~on | (pos<=tol),axis=2)
        at_end=np.roll(shared,-1,axis=1) & np.all(~on | (pos>=lengths[:,:,None]-tol),axis=2)
        only_vertex=(side.max(axis=2)<=tol)&(at_start|at_end)
        return strict | only_vertex.any(axis=1)

    def check_block(left,right):
        ia,ib=cap[left],cap[right]
        shared=(ia[:,:,None]==ib[:,None,:]).sum(axis=(1,2))
        # A shared edge of oriented CCW triangles with opposite incidence
        # separates the two interiors. Edge incidence was checked by the caller.
        remaining=shared!=2
        if not np.any(remaining):return
        ia=ia[remaining];ib=ib[remaining]
        a,b=xy[ia],xy[ib]
        if not np.all(certified(a,b,ia,ib)|certified(b,a,ib,ia)):
            raise ValueError('cap triangles overlap or have nonconforming point/edge contact')

    left=[];right=[];count=0
    for idx,a in enumerate(order):
        end=np.searchsorted(x,high[a,0]+tol,side='right')
        b=order[idx+1:end]
        b=b[(low[b,1]<=high[a,1]+tol)&(high[b,1]>=low[a,1]-tol)]
        if len(b):
            left.append(np.full(len(b),a,dtype=np.int64));right.append(b);count+=len(b)
        if count>=8192:
            check_block(np.concatenate(left),np.concatenate(right));left=[];right=[];count=0
    if count:check_block(np.concatenate(left),np.concatenate(right))


def _prepared(points,cap,levels,diagonals):
    p=np.ascontiguousarray(points,dtype=np.float64)
    c=np.asarray(cap); l=np.asarray(levels)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():raise ValueError('finite Nx3 points required')
    if c.dtype.kind not in 'iu' or c.ndim!=2 or c.shape[1]!=3 or not len(c):raise ValueError('integer cap triangles required')
    if l.dtype.kind not in 'iu' or l.ndim!=2 or l.shape[0]<2 or l.shape[1]<3:raise ValueError('integer matching cap levels required')
    if l.min()<0 or l.max()>=len(p) or l.size!=len(p) or not np.all(np.bincount(l.ravel().astype(np.int64),minlength=len(p))==1):
        raise ValueError('layer IDs must partition all points without duplicates')
    if c.min()<0 or c.max()>=l.shape[1] or len(np.unique(c))!=l.shape[1]:raise ValueError('invalid or unused cap IDs')
    c=np.ascontiguousarray(c,dtype=np.int64);l=np.ascontiguousarray(l,dtype=np.int64)
    xy=p[l[0],:2]
    if len(np.unique(xy,axis=0))!=len(xy):raise ValueError('coincident cap coordinates')
    heights=p[l[:,0],2]
    if np.any(np.diff(heights)<=0):raise ValueError('layers must strictly increase')
    for z,ids in zip(heights,l):
        if not np.array_equal(p[ids,:2],xy) or not np.all(p[ids,2]==z):
            raise ValueError('every layer must be flat and match the cap coordinates')
    t=xy[c]
    u,v=t[:,1]-t[:,0],t[:,2]-t[:,0]
    area2=u[:,0]*v[:,1]-u[:,1]*v[:,0]
    eps=128*np.finfo(float).eps
    if np.any(area2<=eps*np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1)):
        raise ValueError('cap triangle is inverted or degenerate')
    min_volume=area2.min()*np.diff(heights).min()/6
    max_volume=area2.max()*np.diff(heights).max()
    if not np.isfinite(max_volume) or max_volume>=np.finfo(float).max/3 or min_volume<=np.finfo(float).tiny:
        raise ValueError('unsupported prism volume range')
    # Wall choice per oriented cap edge: lower first endpoint -> upper second.
    edges=np.stack((c,np.roll(c,-1,axis=1)),axis=2)
    natural=edges[:,:,0]<edges[:,:,1]
    if diagonals is None:diagonals=natural
    diagonals=np.asarray(diagonals)
    if diagonals.shape!=c.shape or diagonals.dtype.kind!='b':raise ValueError('expected three Boolean diagonals per cap triangle')
    keys=np.sort(edges.reshape(-1,2),axis=1)
    _,inv,n=np.unique(keys,axis=0,return_inverse=True,return_counts=True);inv=inv.ravel()
    if np.any(n>2) or np.any(np.bincount(inv,weights=np.where(natural.ravel(),1,-1))[n==2]):
        raise ValueError('nonmanifold or inconsistently oriented cap edges')
    boundary_degree=np.bincount(keys[n[inv]==1].ravel(),minlength=len(xy))
    if np.any((boundary_degree!=0)&(boundary_degree!=2)):
        raise ValueError('nonmanifold cap boundary vertex')
    agreed=np.bincount(inv,weights=(diagonals==natural).ravel())
    if np.any((agreed!=0)&(agreed!=n)):raise ValueError('adjacent fronts prescribe incompatible wall diagonals')
    _check_planar_cap(xy,c)
    signs=np.zeros((len(c),2),dtype=np.uint32)
    signs[:]=_mask(((0,2,1),(3,4,5)))
    for i,j in ((0,1),(1,2),(2,0)):
        m1=_mask(((i,j,j+3),(i,j+3,i+3)))
        m0=_mask(((i,j,i+3),(j,j+3,i+3)))
        signs|=np.where(diagonals[:,i,None],m1,m0).astype(np.uint32)
    return p,c,l,np.ascontiguousarray(signs),area2,heights


def _cpu_batch(p,c,l,signs,area2,heights):
    count=len(c)*(len(l)-1); output=np.empty((count,3,4),dtype=np.int64)
    plus,minus=_ORIENTED.T; all_faces=plus|minus
    for start in range(0,count,4096):
        end=min(start+4096,count)
        k,i=np.divmod(np.arange(start,end),len(c))
        ids=np.column_stack((l[k[:,None],c[i]],l[k[:,None]+1,c[i]]))
        coords=p[ids[:,_CANDIDATES]]
        a,b,d=(coords[:,:,n]-coords[:,:,0] for n in (1,2,3))
        signed6=np.einsum('nmi,nmi->nm',np.cross(a,b),d)
        expected=area2[i]*(heights[k+1]-heights[k])
        # Each non-flat four-vertex subset spans one third of its vertical prism.
        # Certify that fact on the actual coordinates, not just in the catalogue.
        positive=np.abs(signed6-expected[:,None])<=1e-10*expected[:,None]
        pos,neg=signs[i].T.copy(); initial=pos|neg; signed_sum=np.zeros(end-start)
        eligible=positive & (((all_faces & _EXTERIOR)[None,:] & ~initial[:,None])==0)
        rows=np.arange(end-start)
        for step in range(3):
            active=pos|neg
            shared=(pos[:,None]&plus)|(neg[:,None]&minus)
            wrong=(pos[:,None]&minus)|(neg[:,None]&plus)
            consumed=(all_faces & _EXTERIOR)[None,:] & ~active[:,None]
            legal=eligible & (shared!=0) & (wrong==0) & (consumed==0)
            if not np.all(legal.any(axis=1)):raise ValueError('prism has no legal advancing-front candidate')
            choice=np.argmax(legal,axis=1)
            eligible[rows,choice]=False
            output[start:end,step]=ids[rows[:,None],_CANDIDATES[choice]]
            signed_sum+=signed6[rows,choice]
            pos=(pos&~all_faces[choice])|(minus[choice]&~active)
            neg=(neg&~all_faces[choice])|(plus[choice]&~active)
        if np.any(pos|neg) or np.any(abs(signed_sum-3*expected)>1e-10*3*expected):
            raise ValueError('prism failed oriented closure or volume certificate')
    return output.reshape(-1,4)


@lru_cache(maxsize=16)
def _cuda_kernel(device_id):
    import cupy as cp
    def initializer(table):
        return '{'+','.join('{'+','.join(map(str,row))+'}' for row in table.tolist())+'}'
    code=(Path(__file__).with_name('batched_prism.cu').read_text(encoding='utf8')
          .replace('CANDIDATE_TABLE',initializer(_CANDIDATES))
          .replace('MASK_TABLE',initializer(_ORIENTED)).replace('EXTERIOR_BITS',str(_EXTERIOR)+'u'))
    return cp.RawKernel(code,'batch_front',options=('--std=c++11',))


def advance_batched_prisms(points,cap,levels,backend='cuda',*,diagonals=None):
    """Return host int64 tetrahedra and diagnostics for a previously validated cap.

    levels[k,i] is the global point ID for cap node i at height k. Faces in
    cap are CCW. Optional wall diagonals are per cap face and oriented cap edge.
    Unfillable/degenerate prisms raise before a mesh is returned.
    """
    start=perf_counter()
    p,c,l,signs,area2,heights=_prepared(points,cap,levels,diagonals)
    engine=prism_backend(backend,len(c)*(len(l)-1))
    count=len(c)*(len(l)-1);begin=perf_counter()
    if engine.name=='cuda':
        cp=engine.cp
        with engine.device:
            output=cp.empty((count,3,4),dtype=cp.int64);failed=cp.zeros(1,dtype=cp.uint32)
            args=(cp.asarray(p),cp.asarray(c),cp.asarray(l),cp.asarray(signs),cp.asarray(area2),
                  cp.asarray(heights),np.int64(len(c)),np.int64(l.shape[1]),np.int64(count),output,failed)
            kernel=_cuda_kernel(engine.device.id)
            kernel(((count+127)//128,),(128,),args)
            if cp.asnumpy(failed)[0]:raise ValueError('prism has no legal candidate or failed closure/volume certificate')
            tets=cp.asnumpy(output).reshape(-1,4)
    elif engine.name=='cpu':tets=_cpu_batch(p,c,l,signs,area2,heights)
    else:raise ValueError('expected cpu or cuda backend')
    end=perf_counter(); launches=int(engine.name=='cuda')
    return tets,{'prisms_meshed':count,'front_iterations':3*count,
                 'proposals':15*count,'candidate_geometry_evaluations':15*count,
                 'front_eligibility_tests':45*count,'remaining_front_faces':0,'cpu_closure_prisms':0,
                 'kernel_calls':{'batch_front':launches},'calls':{'batch_front':1},
                 'timings_seconds':{'batch_front':end-begin},'total_seconds':end-start,
                 'input_contract':'straight vertical prisms over an embedded oriented cap',
                 'validation':'per-prism positive determinant, wall-compatibility, dynamic oriented closure and volume certificate'}
