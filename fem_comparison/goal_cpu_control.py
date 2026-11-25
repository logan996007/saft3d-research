"""Freeze an optimized U mesh's nodes and oriented boundary, regenerate CPU interior."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .audit_goal_u import audit
from .cpu_meshes import extract_boundary
from .refined_cpu import degrees, oriented_triangle_keys, tetrahedralize


def _tet_keys(cells):
    result=np.sort(cells,axis=1)
    return result[np.lexsort(result[:,::-1].T)]


def generate(mesh,out):
    mesh,out=Path(mesh),Path(out)
    with np.load(mesh,allow_pickle=False) as data:
        points=data['vertices'];faces=data['boundary_faces'];source_cells=data['tetrahedra']
    if len(np.unique(source_cells))!=len(points) or degrees(points,'u')!=3078:
        raise ValueError('expected fully used U-bracket vertices and 3078 free displacement DOFs')
    prescribed=oriented_triangle_keys(faces)
    flags='pYYMQzO0S0'
    attempts=[]
    # Selection depends only on preserving all prescribed coordinates and
    # oriented surface faces. The loading and displacement are evaluated later.
    for seed in (None,*range(12)):
        order=(np.arange(len(points)) if seed is None else np.random.default_rng(seed).permutation(len(points)))
        try:
            p,t=tetrahedralize(points[order],np.argsort(order)[faces],flags)
        except (RuntimeError,ValueError) as exc:
            attempts.append({'seed':seed,'error':str(exc)})
            continue
        same_points=np.array_equal(p,points[order])
        attempts.append({'seed':seed,'output_node_count':len(p),'bitwise_equal_points':same_points})
        if not same_points:continue
        cells=order[t]
        same_boundary=oriented_triangle_keys(extract_boundary(cells))==prescribed
        attempts[-1]['exact_oriented_boundary']=same_boundary
        if same_boundary: break
    else:
        raise RuntimeError(f'CPU TetGen did not preserve every node and boundary facet: {attempts}')
    out.mkdir(parents=True,exist_ok=True)
    result=out/'mesh.npz'
    if result.resolve()==mesh.resolve():raise ValueError('refusing to overwrite the supplied optimization mesh')
    existing=out/'audit.json'; reuse=False
    if existing.exists() and result.exists():
        old_report=json.loads(existing.read_text(encoding='utf8'))
        with np.load(result,allow_pickle=False) as prior:
            reuse=(old_report.get('sha256')==hashlib.sha256(result.read_bytes()).hexdigest()
                   and old_report.get('mesh_validation',{}).get('valid') is True
                   and old_report.get('p1',{}).get('free_dofs')==3078
                   and np.array_equal(prior['vertices'],points)
                   and oriented_triangle_keys(prior['boundary_faces'])==prescribed
                   and np.array_equal(_tet_keys(prior['tetrahedra']),_tet_keys(cells)))
    if not reuse:
        np.savez_compressed(result,vertices=points,tetrahedra=cells,boundary_faces=faces)
        audit(result,out)
    report=json.loads(existing.read_text(encoding='utf8'))
    if report['p1']['free_dofs']!=3078 or not report['mesh_validation']['valid']:
        raise RuntimeError('CPU control failed the independent validation')
    generation={'method':'CPU TetGen on all optimized coordinates and boundary triangles',
                'source':str(mesh),'source_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),
                'control_sha256':hashlib.sha256(result.read_bytes()).hexdigest(),'switches':flags,
                'all_nodes_preserved':True,'oriented_boundary_preserved':True,'attempts':attempts,
                'reused_audit_of_identical_cells':bool(reuse),'tetrahedra':len(cells),'free_dofs':3078,
                'mean_loaded_ux_m':report['p1']['mean_loaded_ux_m']}
    (out/'generation.json').write_text(json.dumps(generation,indent=2),encoding='utf8')
    print(json.dumps(generation),flush=True)
    return generation


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();generate(args.mesh,args.out)
