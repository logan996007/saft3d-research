"""Generate an error-adapted U-bracket seed at exactly 3,078 P1 free DOFs.

Choose --quick --pilot p1 to reproduce the strict-budget experiment.
Run --verify-p1-aft after generation to verify every prism independently.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from saft3d.u_anisotropic import recover_curvature,recover_p1_curvature,adapted_section,extrude_section
from fem_comparison.elasticity import solve
from fem_comparison.run import ROOT,vtk_fields


def run(quick=False,pilot='p2'):
    folder='p1-anisotropic' if pilot=='p1' else 'anisotropic'
    out=ROOT/'output'/'u-goal-adapt'/folder
    out.mkdir(parents=True,exist_ok=True)
    src=ROOT/'output/local-refinement/u/level2.npz'
    with np.load(src) as d:v,t,f=d['vertices'],d['tetrahedra'],d['boundary_faces']
    centres,G,vol,meta=(recover_p1_curvature(v,t,f) if pilot=='p1' else recover_curvature(v,t))
    np.savez_compressed(out/'pilot.npz',centres=centres,gram=G,volume=vol)
    records=[]
    iterations,resolution=(20,46) if quick else (35,66)
    layouts=[(2,.8,3),(3,.8,3),(6,.8,3),(3,1.,5),(2,1.,5),(3,.55,2)]
    for nz,expo,aspect in layouts:
        xy,cap,nb=adapted_section(centres,G,vol,1026//nz,exponent=expo,aspect=aspect,iterations=iterations,resolution=resolution)
        p,t,f,gen=extrude_section(xy,cap,nb,nz,backend='cuda')
        q,_=solve(p,t,f)
        if q['free_dofs']!=3078: raise ValueError('unconstrained DOF budget changed')
        label=f'z{nz}_weight{expo}_aspect{aspect}'
        np.savez_compressed(out/(label+'.npz'),vertices=p,tetrahedra=t,boundary_faces=f)
        parameters={'size_exponent':expo,'aspect_limit':aspect,'CVT_iterations':iterations,'quadrature_samples_per_m':resolution,'seed':3141}
        rec=dict(label=label,generation=gen,elasticity=q,estimator=meta,parameters=parameters,
                 pilot_source='output/local-refinement/u/level2.npz',pilot_sha256=hashlib.sha256(src.read_bytes()).hexdigest())
        records.append(rec)
        (out/'trials.json').write_text(json.dumps(records,indent=2));print(label,q['mean_loaded_ux_m']*1e3,flush=True)
        # This trial's quality criterion depends only on the current P1 solve.
        if q['compliance_J']==max(row['elasticity']['compliance_J'] for row in records):
            filename='seed' if pilot=='p1' else 'best'
            np.savez_compressed(out/(filename+'.npz'),vertices=p,tetrahedra=t,boundary_faces=f)
            (out/(filename+'.json')).write_text(json.dumps(rec,indent=2))
    filename='seed' if pilot=='p1' else 'best'
    with np.load(out/(filename+'.npz')) as d:p,t,f=d['vertices'],d['tetrahedra'],d['boundary_faces']
    _,fields=solve(p,t,f)
    vtk_fields(out/(filename+'.vtk'),p,t,{'displacement_m':fields['displacement']},{'von_mises_Pa':fields['von_mises']})

def verify_p1_aft():
    from time import perf_counter
    from saft3d.thickness_front import _advance_with_local_aft
    from fem_comparison.cpu_meshes import extract_boundary
    out=ROOT/'output/u-goal-adapt'
    folder=out/'p1-anisotropic'
    with np.load(folder/'seed.npz') as data:
        points,cells,boundary=[data[k] for k in ('vertices','tetrahedra','boundary_faces')]
    planes=json.loads((folder/'seed.json').read_text())['generation']['z_planes']
    count=len(points)//planes
    cap=np.ascontiguousarray(boundary[(boundary<count).all(axis=1)][:,[0,2,1]],dtype=np.int64)
    levels=np.arange(len(points),dtype=np.int64).reshape(planes,count)
    start=perf_counter()
    generated,meta=_advance_with_local_aft(points,cap,levels,'cuda')
    signature=lambda faces:set(map(tuple,np.sort(faces,axis=1)))
    if len(generated)!=len(cells) or signature(generated)!=signature(cells):
        raise RuntimeError('AFT did not reproduce the seed tetrahedra')
    if signature(extract_boundary(generated))!=signature(boundary):
        raise RuntimeError('AFT did not reproduce the seed boundary facets')
    result={'seed_file':str((folder/'seed.npz').relative_to(ROOT)),
            'sha256':hashlib.sha256((folder/'seed.npz').read_bytes()).hexdigest(),
            'method':'Each six-vertex prism and its oriented triangulated sidewalls are passed to the general CUDA advancing front.',
            'identical_tetrahedron_keys':True,'identical_boundary_triangle_keys':True,
            'tet_count':len(cells),'vertices':len(points),'aft_statistics':meta,
            'elapsed_seconds':perf_counter()-start}
    (out/'p1-verified-aft.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(folder/'actual-aft-seed.npz',vertices=points,tetrahedra=generated,boundary_faces=boundary)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true');parser.add_argument('--pilot',choices=['p1','p2'],default='p2')
    parser.add_argument('--verify-p1-aft',action='store_true')
    args=parser.parse_args()
    if args.verify_p1_aft:verify_p1_aft()
    else:run(args.quick,args.pilot)
