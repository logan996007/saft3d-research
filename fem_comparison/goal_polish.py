"""Alternate variational connectivity swaps and FE-driven vertex relocation."""
import argparse
import json
from pathlib import Path
import numpy as np
from .run import vtk_fields
from .elasticity import solve
from saft3d.compliance_flips import energy_flips
from saft3d.compliance_relocation import relocate


def run(source,out,cycles,r_steps,flip_steps,weight):
    out.mkdir(parents=True,exist_ok=True)
    with np.load(source) as m:p,t,f=[m[k] for k in ('vertices','tetrahedra','boundary_faces')]
    metrics,fields=solve(p,t,f)
    if metrics['free_dofs']!=3078:raise ValueError('Requires 3078 free P1 DOFs')
    history=[]
    for cycle in range(cycles):
        t,metrics,fields,swaps=energy_flips(p,t,f,flip_steps)
        p,metrics,fields,moves=relocate(p,t,f,r_steps,weight)
        if metrics['free_dofs']!=3078:raise ValueError('Budget changed')
        history.append({'cycle':cycle+1,'swaps':swaps,'relocation':moves})
        np.savez_compressed(out/'mesh.npz',vertices=p,tetrahedra=t,boundary_faces=f)
        print(cycle+1,metrics['mean_loaded_ux_m']*1000,'mm',flush=True)
        (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf8')
    metrics.update(source=str(source),cycles=cycles,relocation_steps=r_steps,flip_steps=flip_steps,
                   weight=weight,algorithm='P1 elastic energy patch swaps and shape derivative relocation')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    vtk_fields(out/'field.vtk',p,t,{'displacement_m':fields['displacement']},{'von_mises_Pa':fields['von_mises']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('out',type=Path)
    parser.add_argument('--cycles',type=int,default=2)
    parser.add_argument('--steps',type=int,default=36)
    parser.add_argument('--flips',type=int,default=5)
    parser.add_argument('--weight',type=float,default=.003)
    args=parser.parse_args()
    run(args.source,args.out,args.cycles,args.steps,args.flips,args.weight)
