"""Reproduce the fixed-budget compliance shape optimization experiment."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from saft3d.compliance_relocation import relocate,check_boundary_motion
from saft3d.validation import validate_mesh
from .run import ROOT, vtk_fields


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'output/local-refinement/u/level2.npz')
    parser.add_argument('--out',type=Path,default=ROOT/'output/u-goal-adapt/relocate-barrier')
    parser.add_argument('--iterations',type=int,default=200)
    parser.add_argument('--weight',type=float,default=.004)
    args=parser.parse_args();out=args.out;out.mkdir(parents=True,exist_ok=True)
    data=np.load(args.source);p,t,f=[data[k] for k in ('vertices','tetrahedra','boundary_faces')]
    start=perf_counter()
    def checkpoint(points,history):
        if history[-1]['iteration']%10==0:
            print(history[-1],flush=True)
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
    p,met,fields,history=relocate(p,t,f,args.iterations,args.weight,checkpoint)
    if met['free_dofs']!=3078:raise ValueError('DOF budget changed')
    np.savez_compressed(out/'mesh.npz',vertices=p,tetrahedra=t,boundary_faces=f)
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
    reference=json.loads((ROOT/'output/fem-comparison/results.json').read_text())['independent_elasticity']['u_finer_p2']['mean_loaded_ux_m']
    met.update(domain_check=check_boundary_motion(data['vertices'],p,f),
               reference_m=reference,relative_ux_error=abs(met['mean_loaded_ux_m']/reference-1),
               iterations=len(history)-1,elapsed_before_validation=perf_counter()-start,
               algorithm='P1 compliance shape derivative and regularized L-BFGS relocation',
               source=str(args.source),regularization=args.weight,iteration_budget=args.iterations)
    print('FINAL',met,flush=True)
    check=validate_mesh(p,t,f)
    if not check['valid']:raise RuntimeError(check['errors'])
    met['validation']=check
    (out/'metrics.json').write_text(json.dumps(met,indent=2),encoding='utf8')
    vtk_fields(out/'field.vtk',p,t,{'displacement_m':fields['displacement']},{'von_mises_Pa':fields['von_mises']})

if __name__=='__main__': main()
