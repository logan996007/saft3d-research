"""Geometry-only CPU TetGen controls at prescribed numbers of free P1 DOFs.

python -m fem_comparison.refined_cpu
python -m fem_comparison.refined_cpu --same-nodes <u|s> <mesh.npz>
"""
import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import tetgen

from saft3d.general_surfaces import make_general_surface
from fem_comparison.cpu_meshes import extract_boundary, audit_boundary

OUT = Path(__file__).resolve().parents[1] / 'output' / 'local-refinement' / 'cpu'
SHAPES = {'u': 'u_prism', 's': 'slender_s'}


def fixed_nodes(points, shape):
    if shape == 'u':
        return np.isclose(points[:, 1], 3, rtol=0, atol=1e-10) & (points[:, 0] <= 1+1e-10)
    return np.isclose(abs(points[:, 2]), .12, rtol=0, atol=1e-10)


def degrees(points, shape):
    return (3 if shape == 'u' else 1) * int(np.count_nonzero(~fixed_nodes(points,shape)))


def oriented_triangle_keys(faces):
    # Cyclic permutations represent the same oriented triangle; reversal does not.
    return {min((a,b,c),(b,c,a),(c,a,b)) for a,b,c in faces}


def tetrahedralize(points, triangles, flags):
    markers = np.arange(1, len(triangles)+1, dtype=np.int32)
    gen = tetgen.TetGen(np.ascontiguousarray(points), np.ascontiguousarray(triangles,np.int32), markers)
    v, t, _, _ = gen.tetrahedralize(switches=flags)
    used, inv = np.unique(t, return_inverse=True)
    v, t = v[used], inv.reshape(-1,4)
    xyz = v[t]
    det = np.linalg.det(xyz[:,1:]-xyz[:,:1]); inverted=det<0
    t[inverted]=t[inverted][:,[1,0,2,3]]
    if np.min(abs(det))<1e-14:
        raise ValueError('degenerate TetGen cell')
    return v,t


def generate_matched(shape, target):
    """Choose volume and Steiner budgets by DOF counts, without evaluating fields."""
    if target <= 0 or (shape=='u' and target%3):
        raise ValueError('unattainable P1 DOF budget')
    original, faces, meta = make_general_surface(SHAPES[shape])
    node_budget = target//3 if shape=='u' else target
    # Estimate a uniform target volume. Search the volume criterion first;
    # unbounded quality refinement alone can exceed thermal DOF budgets.
    volume = meta['analytic_volume'] / (4*node_budget)
    trials=[]
    def sample(vol,limit=None):
        flags = f'pMQzq1.8a{vol:.9f}' + ('' if limit is None else f'S{limit}')
        pts, tet=tetrahedralize(original, faces, flags)
        n=degrees(pts,shape)
        trials.append({'flags':flags, 'free_dofs':n})
        return n,pts,tet,flags
    sample0=sample(volume)
    if sample0[0]==target: return *sample0[1:],trials
    above,below=(volume,None) if sample0[0]>target else (None,volume)
    for _ in range(10):
        volume=volume*1.8 if above is not None else volume/1.8
        attempt=sample(volume)
        if attempt[0]==target: return *attempt[1:],trials
        if attempt[0]>target: above=volume
        else: below=volume
        if above is not None and below is not None:break
        if volume>meta['analytic_volume']:break
    if above is None: raise RuntimeError('failed to bracket requested number of DOFs')
    if below is not None:
        for _ in range(9):
            volume=(above+below)/2
            attempt=sample(volume)
            if attempt[0]==target: return *attempt[1:],trials
            if attempt[0]>target:above=volume
            else:below=volume
    # Limit the native queue of Steiner insertions. Boundary vertices and
    # constrained DOFs are counted separately, unlike a total-point budget.
    unbounded=sample(above); hi=len(unbounded[1])-len(original);lo=0
    cache={}
    def capped(limit):
        if limit not in cache:cache[limit]=sample(above,limit)
        return cache[limit]
    while lo<=hi:
        mid=(lo+hi)//2;attempt=capped(mid)
        if attempt[0]==target: return *attempt[1:],trials
        if attempt[0]<target:lo=mid+1
        else:hi=mid-1
    for cap in range(max(0,lo-24), min(len(unbounded[1]),lo+24)+1):
        attempt=capped(cap)
        if attempt[0]==target: return *attempt[1:],trials
    # Finish an occasional skipped native budget by splitting largest-volume
    # cells at their barycentres. Every such conforming split adds one free node.
    feasible=[x for x in cache.values() if x[0]<target]
    if not feasible:raise RuntimeError('no mesh below requested DOFs')
    n,pts,tet,flags=max(feasible,key=lambda x:x[0]);missing=(target-n)//(3 if shape=='u' else 1)
    face_indices=np.array([(1,2,3),(0,3,2),(0,1,3),(0,2,1)])
    for _ in range(missing):
        xyz=pts[tet];i=int(np.argmax(np.linalg.det(xyz[:,1:]-xyz[:,:1])))
        apex=len(pts);pts=np.vstack((pts,xyz[i].mean(axis=0)))
        children=np.column_stack((np.full(4,apex),tet[i][face_indices]))
        tet=np.vstack((np.delete(tet,i,axis=0),children))
    return pts,tet,flags+f' + {missing} barycentre splits',trials


def save_case(name,shape,v,t,flags,method,trials=None,source=None,expected_boundary=None):
    original,faces,meta=make_general_surface(SHAPES[shape]);f=extract_boundary(t)
    xyz=v[t];vols=np.linalg.det(xyz[:,1:]-xyz[:,:1])/6
    if min(vols)<=0 or abs(vols.sum()/meta['analytic_volume']-1)>1e-10:raise ValueError('invalid volume')
    # Independently generated controls preserve original facet partitions. A
    # layered surface can legally re-triangulate a plane, so its same-node
    # control instead verifies its exact prescribed surface and prism volume.
    if expected_boundary is None:
        area_error=audit_boundary(v,f,original,faces)
    else:
        if oriented_triangle_keys(f)!=oriented_triangle_keys(expected_boundary):
            raise ValueError('CPU changed prescribed boundary triangles or orientation')
        area_error=0.
    edge_sq=sum(np.sum((xyz[:,a]-xyz[:,b])**2,axis=1) for a,b in combinations(range(4),2))
    q=12*(3*vols)**(2/3)/edge_sq
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(OUT/f'{name}.npz',vertices=v,tetrahedra=t,boundary_faces=f)
    info={'name':name,'shape':shape,'method':method,'tetgen_version':tetgen.__version__,'flags':flags,
          'free_dofs':degrees(v,shape),'nodes':len(v),'fixed_nodes':int(fixed_nodes(v,shape).sum()),
          'boundary_nodes':len(np.unique(f)),'tetrahedra':len(t),'boundary_triangles':len(f),
          'volume_m3':float(vols.sum()),'analytic_volume_m3':meta['analytic_volume'],
          'max_boundary_facet_area_relative_error':area_error,'min_mean_ratio':float(q.min()),
          'median_mean_ratio':float(np.median(q)),'input_npz':source,'dof_search':trials or []}
    (OUT/f'{name}.json').write_text(json.dumps(info,indent=2),encoding='utf8')
    print(name,info['free_dofs'],info['nodes'],info['tetrahedra'],flags,flush=True)
    return info


def make_native(shape,target):
    v,t,flags,trials=generate_matched(shape,target)
    if degrees(v,shape)!=target: raise RuntimeError('budget mismatch')
    return save_case(f'{shape}-{target}',shape,v,t,flags,'independent CPU TetGen quality refinement',trials=trials)


def make_same_nodes(shape, path, tag='same'):
    with np.load(path) as data:
        input_points=data['vertices']; triangles=data['boundary_faces']; tet=data['tetrahedra']
    used=np.unique(tet); p=input_points[used]; faces=np.searchsorted(used,triangles)
    flags='pYYMQzO0S0'
    # Co-spherical layer nodes admit multiple legal tetrahedralizations. If
    # PLC recovery adds a node, try a deterministic input permutation. Accept
    # the first result preserving every coordinate and boundary face, with no
    # field solve or quality score used to select the permutation.
    trials=[]
    for seed in (None,*range(12)):
        order=(np.arange(len(p)) if seed is None else np.random.default_rng(seed).permutation(len(p)))
        v,t=tetrahedralize(p[order],np.argsort(order)[faces],flags)
        trials.append({'permutation_seed':seed,'output_nodes':len(v),'input_nodes':len(p)})
        if not np.array_equal(v,p[order]): continue
        t=order[t]
        if oriented_triangle_keys(extract_boundary(t))!=oriented_triangle_keys(faces):continue
        return save_case(f'{shape}-{degrees(p,shape)}-{tag}',shape,p,t,flags,'CPU TetGen on prescribed SAFT coordinates',
                         source=str(path),expected_boundary=faces,trials=trials)
    raise RuntimeError('Could not preserve all nodes and boundary facets under deterministic input orders')


def audit_saved():
    """Independent geometric verification, including pairwise overlap/contact."""
    from saft3d.validation import validate_mesh
    records={}
    for path in sorted(OUT.glob('*.npz'),key=lambda x:x.stat().st_size):
        with np.load(path) as data:
            inputs=[data[k] for k in ('vertices','tetrahedra','boundary_faces')]
        result=validate_mesh(*inputs)
        if not result['valid']:raise ValueError((path.name,result['errors']))
        fields=('valid','tet_count','checked_pairs','lp_calls','overlap_count',
                'nonconforming_contact_count','relative_volume_error','boundary_faces_match','validation_seconds')
        records[path.stem]={key:result[key] for key in fields}
    (OUT/'audit.json').write_text(json.dumps(records,indent=2),encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',nargs=2,metavar=('SHAPE','FREE_DOFS'))
    parser.add_argument('--same-nodes',nargs=2,metavar=('SHAPE','MESH_NPZ'))
    parser.add_argument('--tag',default='same',choices=('same','wall-same'))
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    if args.audit:audit_saved();return
    if args.same_nodes:make_same_nodes(args.same_nodes[0],Path(args.same_nodes[1]),args.tag);return
    jobs=[(args.case[0],int(args.case[1]))] if args.case else [
        ('u',156),('u',966),('u',3078),('s',48),('s',96),('s',192)]
    for shape,target in jobs: make_native(shape,target)


if __name__=='__main__': main()
