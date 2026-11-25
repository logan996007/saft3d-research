"""Thickness-directed tetrahedral front for a straight, z-extruded PLC.

An oriented cap triangulation advances between prescribed z levels. Each cap
triangle owns a triangular prism. Its six boundary vertices and consistently
triangulated interfaces are sent to the ordinary local AFT tetrahedralizer.
All prism interiors are meshed independently. An optional direct sweep can
be used as a control on the same geometry.
"""
from collections import Counter, defaultdict

import numpy as np
from time import perf_counter

from .backend import create_backend, prism_backend
from .mesher import check_surface
from .general_mesher import AdvancingFrontConfig,grow_general_mesh
from .surface_validation import _check_vertex_links
from scipy.spatial import cKDTree


_ADVANCE_PRISMS = r'''
extern "C" __global__ void advance_prisms(const long long* triangles,
    const long long* lower, const long long* upper, const double* xy,
    long long count, long long* out) {
    long long i=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=count) return;
    long long a=triangles[3*i], b=triangles[3*i+1], c=triangles[3*i+2];
    if(a>b) {long long s=a;a=b;b=s;}
    if(b>c) {long long s=b;b=c;c=s;}
    if(a>b) {long long s=a;a=b;b=s;}
    double det=(xy[2*b]-xy[2*a])*(xy[2*c+1]-xy[2*a+1])
             -(xy[2*b+1]-xy[2*a+1])*(xy[2*c]-xy[2*a]);
    long long cells[12]={lower[a],lower[b],lower[c],upper[c],
                         lower[a],lower[b],upper[c],upper[b],
                         lower[a],upper[a],upper[b],upper[c]};
    for(int t=0;t<3;t++) {
        if(det<0) {long long s=cells[4*t];cells[4*t]=cells[4*t+1];cells[4*t+1]=s;}
        for(int j=0;j<4;j++) out[12*i+4*t+j]=cells[4*t+j];
    }
}
'''


def _cap_template(vertices, faces):
    """Check the input is a valid extrusion with a matching upper/lower cap."""
    if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all():
        raise ValueError('finite three-dimensional vertices required')
    if faces.ndim!=2 or faces.shape[1]!=3 or faces.dtype.kind not in 'iu':
        raise ValueError('oriented triangular faces with integer indices required')
    extent=np.ptp(vertices,axis=0)
    if np.any(extent<=0): raise ValueError('zero domain extent')
    normalized=(vertices-vertices.min(axis=0))/extent.max()
    check_surface(normalized,faces,1.e-10)
    _check_vertex_links(faces)
    if len(cKDTree(normalized).query_pairs(1.e-10,output_type='ndarray')):
        raise ValueError('surface has coincident or near-coincident nodes')
    low,high=vertices[:,2].min(),vertices[:,2].max()
    bottom_ids=np.flatnonzero(vertices[:,2]==low)
    top_ids=np.flatnonzero(vertices[:,2]==high)
    if len(bottom_ids)+len(top_ids)!=len(vertices) or len(top_ids)!=len(bottom_ids):
        raise ValueError('expected straight extrusion with vertices on two caps')
    xy=vertices[bottom_ids,:2].copy()
    top_by_xy={tuple(vertices[i,:2]):i for i in top_ids}
    if set(top_by_xy)!=set(map(tuple,xy)):
        raise ValueError('upper and lower cap coordinates must match vertically')
    matched_top=np.array([top_by_xy[tuple(node)] for node in xy],dtype=np.int64)
    local=np.empty(len(vertices),np.int64)
    local[bottom_ids]=local[matched_top]=np.arange(len(xy))
    fz=vertices[faces,2]
    bottom=local[faces[(fz==low).all(axis=1)]][:,::-1]
    top=local[faces[(fz==high).all(axis=1)]]
    if len(top)==0 or sorted(map(tuple,np.sort(top,axis=1)))!=sorted(map(tuple,np.sort(bottom,axis=1))):
        raise ValueError('upper and lower cap triangulations must match')
    delta=xy[top[:,1:]]-xy[top[:,:1]]
    signed_areas=(delta[:,0,0]*delta[:,1,1]-delta[:,0,1]*delta[:,1,0])/2
    if np.any(signed_areas<=0): raise ValueError('upper cap must point out of domain')
    edges=[(int(tri[i]),int(tri[(i+1)%3])) for tri in top for i in range(3)]
    multiplicity=Counter(tuple(sorted(e)) for e in edges)
    incidence=Counter(edges)
    if any(n>2 or (n==2 and incidence[edge]!=1) for edge,n in multiplicity.items()):
        raise ValueError('inconsistent internal cap edge incidence')
    if len(np.unique(top))!=len(xy):
        raise ValueError('unused cap vertex')
    # A non-overlapping cap times an interval is embedded. Prove that the
    # supplied side triangles cover precisely its oriented perimeter walls.
    from .batched_prism import _check_planar_cap
    _check_planar_cap(normalized[bottom_ids,:2],top)
    side_edges=[e for e in edges if multiplicity[tuple(sorted(e))]==1]
    side_keys={tuple(sorted(e)) for e in side_edges}
    walls=defaultdict(list)
    for tri in faces[~((fz==low).all(axis=1)|(fz==high).all(axis=1))]:
        key=tuple(np.unique(local[tri]))
        if key not in side_keys:raise ValueError('sidewalls must be vertical on the cap perimeter')
        walls[key].append(tuple(map(int,tri)))
    def signature(tris):
        return {min((a,b,c),(b,c,a),(c,a,b)) for a,b,c in tris}
    for a,b in side_edges:
        supplied=walls[tuple(sorted((a,b)))]
        choices=[signature(_wall_triangles(bottom_ids[a],bottom_ids[b],matched_top[a],matched_top[b],direction))
                 for direction in (True,False)]
        if len(supplied)!=2 or signature(supplied) not in choices:
            raise ValueError('sidewall must cover its rectangle with two outward triangles')
    return xy,bottom_ids,matched_top,top,side_edges,float(signed_areas.sum())


def _advance_cap(xy,triangles,levels,backend):
    if backend.name=='cuda':
        cp=backend.cp
        triangles_gpu=cp.asarray(triangles, dtype=cp.int64)
        layers_gpu=cp.asarray(levels); xy_gpu=cp.asarray(xy)
        cells=cp.empty((len(levels)-1,len(triangles),3,4),dtype=cp.int64)
        advance=cp.RawKernel(_ADVANCE_PRISMS,'advance_prisms')
        for layer in range(len(cells)):
            advance(((len(triangles)+127)//128,),(128,),
                    (triangles_gpu,layers_gpu[layer],layers_gpu[layer+1],xy_gpu,
                     np.int64(len(triangles)),cells[layer]))
        return cells.reshape(-1,4)
    sorted_ids=np.sort(triangles,axis=1); a,b,c=sorted_ids.T
    ab,ac=xy[b]-xy[a],xy[c]-xy[a]
    negative=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]<0
    # Prepare local oriented indices only once; even very large layer stacks
    # are filled directly into the host result without a second full copy.
    n=len(xy)
    pattern=np.stack([np.column_stack([a,b,c,n+c]),
                      np.column_stack([a,b,n+c,n+b]),
                      np.column_stack([a,n+a,n+b,n+c])],axis=1)
    pattern[negative]=pattern[negative][:,:,[1,0,2,3]]
    cells=np.empty((len(levels)-1,len(triangles),3,4),dtype=np.int64)
    for i in range(len(cells)):
        np.take(levels[i:i+2].reshape(-1),pattern,out=cells[i],mode='clip')
    return cells.reshape(-1,4)


def _refine_cap(xy,cap,perimeter,steps):
    """Conforming 1-to-4 subdivision of an already validated z section."""
    perimeter=np.asarray(perimeter,dtype=np.int64)
    for _ in range(steps):
        n=len(xy)
        pairs=cap[:,[[0,1],[1,2],[2,0]]]
        keys=pairs.min(axis=2)*n+pairs.max(axis=2)
        unique,inverse=np.unique(keys,return_inverse=True)
        mids=n+inverse.reshape(cap.shape)
        xy=np.vstack((xy,(xy[unique//n]+xy[unique%n])*.5))
        a,b,c=cap.T; ab,bc,ca=mids.T
        cap=np.stack([np.column_stack(row) for row in ((a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca))],axis=1).reshape(-1,3)
        left,right=perimeter.T
        key=np.minimum(left,right)*n+np.maximum(left,right)
        mid=n+np.searchsorted(unique,key)
        perimeter=np.column_stack((left,mid,mid,right)).reshape(-1,2)
    return xy,cap,perimeter


def _sweep_quality(points,cells,engine,chunk=131072):
    """Evaluate every generated cell, keeping at most one coordinate chunk live."""
    if engine.name!='cuda':
        total=0.;minimum_volume=np.inf;minimum_quality=np.inf
        for first in range(0,len(cells),chunk):
            volumes,quality=engine.evaluate(points[cells[first:first+chunk]])
            total+=float(volumes.sum())
            minimum_volume=min(minimum_volume,float(volumes.min()))
            minimum_quality=min(minimum_quality,float(quality.min()))
        return total,minimum_volume,minimum_quality
    cp=engine.cp
    p=cp.asarray(points)
    stats=[]
    for first in range(0,len(cells),chunk):
        xyz=p[cells[first:first+chunk]]
        n=len(xyz)
        volume=cp.empty(n,dtype=cp.float64);quality=cp.empty_like(volume)
        engine._launch('evaluate_tets',n,(xyz,np.int64(n),volume,quality))
        stats.append(cp.stack((volume.sum(),volume.min(),quality.min())))
    values=cp.stack(stats)
    metric=cp.asnumpy(cp.stack((values[:,0].sum(),values[:,1].min(),values[:,2].min())))
    return tuple(map(float,metric))


def _sweep_boundary(cap,perimeter,levels):
    ncap=len(cap);nwall=len(perimeter);intervals=len(levels)-1
    boundary=np.empty((2*ncap+2*intervals*nwall,3),dtype=np.int64)
    boundary[:ncap]=levels[0,cap[:,::-1]]
    boundary[ncap:2*ncap]=levels[-1,cap]
    wall=boundary[2*ncap:].reshape(intervals,nwall,2,3)
    a,b=np.asarray(perimeter).T;inc=a<b
    for i,(lo,hi) in enumerate(zip(levels[:-1],levels[1:])):
        la,lb,ha,hb=lo[a],lo[b],hi[a],hi[b]
        wall[i,:,0,:]=np.column_stack((la,lb,np.where(inc,hb,ha)))
        wall[i,:,1,:]=np.column_stack((np.where(inc,la,lb),hb,ha))
    return boundary


def _wall_triangles(a,b,upper_a,upper_b,increasing):
    if increasing: return [(a,b,upper_b),(a,upper_b,upper_a)]
    return [(a,b,upper_a),(b,upper_b,upper_a)]


def _advance_with_local_aft(points,cap,levels,backend):
    cells=[]; kernel_calls=Counter(); calls=Counter(); seconds=Counter(); iterations=0; proposals=0
    for lower,upper in zip(levels[:-1],levels[1:]):
        for tri in cap:
            # The interface prescription defines triangular faces only. The
            # AFT receives no tetrahedron connectivity or reference solution.
            global_ids=np.r_[lower[tri],upper[tri]]
            prism=points[global_ids]
            faces=[(0,2,1),(3,4,5)]
            for i in range(3):
                j=(i+1)%3
                faces.extend(_wall_triangles(i,j,i+3,j+3,tri[i]<tri[j]))
            config=AdvancingFrontConfig(backend=backend,size=float(prism[3,2]-prism[0,2]),
                                        max_new_vertices=0,min_quality=0,completion='none')
            out,tets,run=grow_general_mesh(prism,np.asarray(faces),config)
            if not np.array_equal(out,prism) or run['remaining_front_faces'] or run['closure_tets']:
                raise RuntimeError('local AFT did not close this prism at its prescribed interface')
            cells.append(global_ids[tets]); proposals+=run['proposals'];iterations+=run['iterations']
            kernel_calls.update(run['backend_stats']['kernel_calls'])
            calls.update(run['backend_stats']['calls']); seconds.update(run['backend_stats']['timings_seconds'])
    return np.vstack(cells),{'prisms_meshed':len(cells),'front_iterations':iterations,'proposals':proposals,
                             'cpu_closure_prisms':0,'kernel_calls':dict(kernel_calls),'calls':dict(calls),
                             'timings_seconds':dict(seconds)}


def grow_thickness_front(vertices,faces,fractions,*,backend='cuda',mode='batched',cap_refinements=0):
    """Return points, oriented tets, refined boundary and run metadata.

    Fractions are increasing relative distances from the bottom to the top cap,
    including 0 and 1. The original cap vertices retain their IDs. Straight
    sidewalls may acquire new diagonals within their original rectangular PLC
    patches. Each optional cap_refinements step splits every cap triangle
    into four triangles at shared edge midpoints. Input facet checks run on
    the prescribed PLC before this conforming subdivision. In sweep mode the
    returned connectivity is resident on the host and all cell volumes and
    mean ratios have been evaluated using bounded batches.
    """
    start=perf_counter()
    if mode not in ('aft','sweep','batched'): raise ValueError('mode must be batched, aft or sweep')
    vertices=np.asarray(vertices,dtype=np.float64); faces=np.asarray(faces)
    fractions=np.asarray(fractions,dtype=float)
    if (fractions.ndim!=1 or len(fractions)<2 or not np.isfinite(fractions).all()
            or fractions[0]!=0 or fractions[-1]!=1 or np.any(np.diff(fractions)<=0)):
        raise ValueError('level fractions must increase strictly from 0 to 1')
    if isinstance(cap_refinements,bool) or not isinstance(cap_refinements,(int,np.integer)) or cap_refinements<0:
        raise ValueError('cap_refinements must be a nonnegative integer')
    xy,bottom,top,cap,perimeter,area=_cap_template(vertices,faces)
    original_count=len(xy)
    xy,cap,perimeter=_refine_cap(xy,cap,perimeter,cap_refinements)
    extra=len(xy)-original_count
    bottom=np.r_[bottom,np.arange(len(vertices),len(vertices)+extra)]
    top=np.r_[top,np.arange(len(vertices)+extra,len(vertices)+2*extra)]
    count=len(xy); intervals=len(fractions)-1
    height=float(np.ptp(vertices[:,2])); low=float(vertices[:,2].min())
    levels=np.empty((intervals+1,count),dtype=np.int64)
    levels[0],levels[-1]=bottom,top
    first_interior=len(vertices)+2*extra
    levels[1:-1]=np.arange(first_interior,first_interior+(intervals-1)*count).reshape(-1,count)
    points=np.empty((len(levels)*count,3));points[:len(vertices)]=vertices
    for ids,fraction in zip(levels,fractions):
        points[ids,:2]=xy;points[ids,2]=low+height*fraction
    points[:len(vertices)]=vertices
    prepared=perf_counter()
    engine=prism_backend(backend,len(cap)*intervals) if mode=='batched' else create_backend(backend)
    initialized=perf_counter()
    front_evidence={}
    if mode=='sweep':
        cells=_advance_cap(xy,cap,levels,engine)
        if engine.name=='cuda':engine.cp.cuda.get_current_stream().synchronize()
        advanced=perf_counter()
        total_volume,min_volume,min_quality=_sweep_quality(points,cells,engine)
        measured=perf_counter()
        if engine.name=='cuda':cells=engine.cp.asnumpy(cells)
    else:
        if mode == 'batched':
            from .batched_prism import advance_batched_prisms
            cells,front_evidence=advance_batched_prisms(points,cap,levels,backend=engine)
        else:
            cells,front_evidence=_advance_with_local_aft(points,cap,levels,engine.name)
        advanced=perf_counter()
        volumes,quality=engine.evaluate(points[cells])
        total_volume,min_volume,min_quality=float(volumes.sum()),float(volumes.min()),float(quality.min())
        measured=perf_counter()
    if not min_volume>0 or not np.isclose(total_volume,height*area,rtol=1e-11,atol=0):
        raise RuntimeError('directed front failed the orientation or total-volume check')
    boundary=_sweep_boundary(cap,perimeter,levels)
    finished=perf_counter()
    stats=engine.stats
    for name in ('kernel_calls','calls','timings_seconds'):
        total=Counter(stats[name]);total.update(front_evidence.get(name,{}));stats[name]=dict(total)
    methods={'batched':'batched local advancing fronts on prescribed straight prisms',
             'aft':'serial dispatch of independent local advancing fronts on straight prisms',
             'sweep':'direct staircase sweep on prescribed cap layers'}
    meta={'algorithm':methods[mode],
          'mode':mode,'prism_evidence':front_evidence,
          'backend':engine.name,'device':engine.device_info,
          'backend_stats':stats,'advance_prisms_gpu_launches':intervals if engine.name=='cuda' and mode=='sweep' else 0,
          'cap_triangles_per_launch':len(cap),'thickness_intervals':intervals,'levels_z':(low+height*fractions).tolist(),
          'free_thermal_dofs':(intervals-1)*count,'growth_tets':len(cells),'closure_tets':0,
          'aft_volume_fraction':1.,'remaining_front_faces':0,'section_area_m2':area,
          'min_mean_ratio':min_quality,'thickness_step_max_m':height*float(np.max(np.diff(fractions))),
          'cap_refinements':int(cap_refinements),'total_volume':total_volume,'min_volume':min_volume,
          'stage_seconds':{'input_and_point_layout':prepared-start,'backend_setup':initialized-prepared,
                           'connectivity':advanced-initialized,'cell_checks':measured-advanced,
                           'host_output_and_boundary':finished-measured},
          'meshing_seconds':perf_counter()-start,
          'preserved_geometry':'Original matching triangular caps, exactly planar sidewall patches; sidewall diagonals may change.'}
    return points,cells,np.asarray(boundary,np.int64),meta
