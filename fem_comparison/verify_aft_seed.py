"""Re-mesh each swept triangular prism with the general local AFT, then compare."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from saft3d.backend import create_backend
from saft3d.batched_prism import advance_batched_prisms
from saft3d.thickness_front import _advance_with_local_aft
from saft3d.validation import validate_mesh
from .cpu_meshes import extract_boundary
from .elasticity import solve
from .run import ROOT


def cap_and_layers(points,faces):
    """Derive the 2-D template from external cap faces, never from volume cells."""
    zlevels=np.unique(points[:,2])
    if len(zlevels)<2: raise ValueError('at least two z levels required')
    base=np.flatnonzero(points[:,2]==zlevels[0])
    order={tuple(points[i,:2]):j for j,i in enumerate(base)}
    if len(order)!=len(base): raise ValueError('duplicate xy coordinates on cap')
    layers=np.empty((len(zlevels),len(base)),dtype=np.int64)
    for k,z in enumerate(zlevels):
        ids=np.flatnonzero(points[:,2]==z)
        if len(ids)!=len(order) or {tuple(points[i,:2]) for i in ids} != set(order):
            raise ValueError('nonmatching cap vertices in layer')
        for i in ids:layers[k,order[tuple(points[i,:2])]]=i
    if not np.array_equal(layers[0],base):raise RuntimeError('inconsistent global IDs')
    top=faces[(points[faces,2]==zlevels[-1]).all(axis=1)]
    cap=np.array([[order[tuple(points[i,:2])] for i in tri] for tri in top],dtype=np.int64)
    xy=points[base,:2]
    ab=xy[cap[:,1]]-xy[cap[:,0]];ac=xy[cap[:,2]]-xy[cap[:,0]]
    if np.any(ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]<=0):raise ValueError('top-cap orientation mismatch')
    return cap,layers


def unordered_tet_keys(t):
    arr=np.sort(t,axis=1)
    return arr[np.lexsort(arr[:,::-1].T)]


def oriented_triangle_keys(f):
    idx=f.argmin(axis=1)
    cyc=np.stack([f[np.arange(len(f)),(idx+i)%3] for i in range(3)],axis=1)
    return cyc[np.lexsort(cyc[:,::-1].T)]


def verify(mesh_path,out,*,mode='batched',backend='cuda'):
    start=perf_counter()
    with np.load(mesh_path,allow_pickle=False) as seed:
        p=seed['vertices']; prescribed=seed['boundary_faces']
    generation_start=perf_counter()
    cap,layers=cap_and_layers(p,prescribed)
    engine=create_backend(backend)
    # Only nodes, triangular cap and indexing/perimeter rules enter AFT.
    if mode=='batched':
        actual,run=advance_batched_prisms(p,cap,layers,engine)
    elif mode=='serial':
        actual,run=_advance_with_local_aft(p,cap,layers,engine.name)
    else:
        raise ValueError('select batched or serial')
    generation_s=perf_counter()-generation_start
    real_boundary=extract_boundary(actual)
    with np.load(mesh_path,allow_pickle=False) as seed:expected=seed['tetrahedra']
    identical=np.array_equal(unordered_tet_keys(actual),unordered_tet_keys(expected))
    matching_boundary=np.array_equal(oriented_triangle_keys(real_boundary),oriented_triangle_keys(prescribed))
    if not identical or not matching_boundary:raise RuntimeError('AFT and sweep differ in cells or oriented boundary')
    fem,_=solve(p,actual,real_boundary)
    if fem['free_dofs']!=3078 or run['cpu_closure_prisms']!=0:raise RuntimeError('wrong provenance or DOFs')
    location=out.with_suffix('.npz');out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(location,vertices=p,tetrahedra=actual,boundary_faces=real_boundary)
    print('AFT reproduced all tets',len(actual),run['prisms_meshed'],perf_counter()-start,flush=True)
    check=validate_mesh(p,actual,prescribed)
    if not check['valid'] or not np.isclose(check['positive_volume_sum'],7.): raise RuntimeError(check['errors'])
    report={'input':str(mesh_path),'input_sha256':hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
            'aft_mesh':str(location),'aft_sha256':hashlib.sha256(location.read_bytes()).hexdigest(),
            'device':engine.device_info,'dispatch_mode':mode,'levels':len(layers),'cap_triangles':len(cap),
            'same_canonical_tets':bool(identical),'same_oriented_boundary':bool(matching_boundary),
            'tet_count':len(actual),'true_local_aft':run,'first_order_elasticity':fem,
            'generation_s':generation_s,'validation':check,'elapsed_s':perf_counter()-start}
    out.write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps({k:report[k] for k in ('levels','cap_triangles','tet_count','same_canonical_tets','elapsed_s')}))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--mode',choices=['batched','serial'],default='batched')
    parser.add_argument('--backend',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--out',type=Path,default=ROOT/'output/u-goal-adapt/verified-aft-seed.json')
    args=parser.parse_args()
    verify(args.mesh,args.out,mode=args.mode,backend=args.backend)
