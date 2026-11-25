"""Reproduce the fixed-budget quality-guarded 3D U-bracket experiment.

    python -m fem_comparison.improve_u_quality

No external reference is passed to any generation or optimization step.
"""
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json

import numpy as np
from saft3d.quality_guard import (
    guarded_flips, joint_quality_compliance, measures,
    quality_guard, reallocate_short_edges,
)
from saft3d.validation import validate_mesh
from fem_comparison.reference import independent_displacement
from fem_comparison.audit_goal_u import check_physical_boundary
from fem_comparison import elasticity
from fem_comparison.run import ROOT,vtk_fields


def run(path,out,loss_budget=.003,*,fast=False):
    """Protect compliance while improving mean ratio and dihedral angle.

    The full benchmark spends portions of one budget across transactional
    connectivity, quality relaxation and mixed quality/compliance steps. The
    fast mode performs only bounded fixed-connectivity quality relaxation.
    """
    if not np.isfinite(loss_budget) or not (0 < loss_budget <= .05):
        raise ValueError('loss budget must lie in (0,.05]')
    started=perf_counter();out.mkdir(parents=True,exist_ok=True)
    with np.load(path,allow_pickle=False) as data:
        p,t,f=[data[k] for k in ('vertices','tetrahedra','boundary_faces')]
    original_node_count=len(p);initial,_=elasticity.solve(p,t,f)
    C0=initial['compliance_J'];before=measures(p,t);times={};history={}
    def timed(name,fn,*args,**kwargs):
        start=perf_counter();result=fn(*args,**kwargs);times[name]=perf_counter()-start;return result
    if fast:
        p,history['fixed_connectivity']=timed('quality_relax',quality_guard,p,t,f,loss_budget=loss_budget)
    else:
        seed_budget=loss_budget*2/3
        t,history['quality_flips']=timed('quality_flips',guarded_flips,p,t,f,C0*(1-seed_budget*.4),goal=.30,max_passes=12)
        partial,_=elasticity.solve(p,t,f)
        remaining=1-C0*(1-seed_budget)/partial['compliance_J']
        p,history['quality_relax']=timed('quality_relax',quality_guard,p,t,f,loss_budget=max(0.,remaining))
        for goal in (.10,.18,.26):
            key=f'joint_{goal:.2f}'
            p,history[key]=timed(key,joint_quality_compliance,p,t,f,initial,goal=goal,loss_budget=seed_budget,iterations=115)
        p,t,f,history['paired_edges']=timed('paired_edges',reallocate_short_edges,p,t,f,C0*(1-loss_budget),quality_target=.16,passes=6)
        p,history['final_joint']=timed('final_joint',joint_quality_compliance,p,t,f,initial,goal=.15,loss_budget=loss_budget,iterations=100)
    optimization_seconds=perf_counter()-started
    met,fields=elasticity.solve(p,t,f);after=measures(p,t)
    if len(p)!=original_node_count or met['free_dofs']!=initial['free_dofs']:
        raise RuntimeError('fixed node or DOF budget lost')
    measured_loss=1-met['compliance_J']/C0
    if measured_loss>loss_budget+1e-10:raise RuntimeError('compliance loss exceeds budget')
    if after['mean_ratio']['min']+1e-12<before['mean_ratio']['min'] or after['dihedral_degrees']['min']+1e-12<before['dihedral_degrees']['min']:
        raise RuntimeError('mesh-quality guard failed')
    geometry=check_physical_boundary(p,f);validation=validate_mesh(p,t,f,check_overlap=True)
    if not validation['valid']:raise RuntimeError(validation['errors'])
    check=independent_displacement(p,t,1)
    if not np.isclose(check['mean_loaded_ux_m'],met['mean_loaded_ux_m'],rtol=1e-8,atol=1.e-14):raise RuntimeError('independent P1 assemblers disagree')
    references=[1.178976015011427,1.18128326]
    record={'method':'guarded local shape/topology improvement' if not fast else 'fixed-connectivity quality relaxation',
            'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'parameters':{'preset':'full' if not fast else 'fast','compliance_loss_budget':loss_budget},
            'before':before,'after':after,'elasticity_before':initial,'elasticity_after':met,'independent_p1':check,
            'validation':validation,'physical_u_boundary':geometry,'loss_budget':loss_budget,'measured_compliance_loss':measured_loss,
            'reference_displacements_mm':references,'posthoc_error_percent':[100*abs(1-met['mean_loaded_ux_m']*1000/ref) for ref in references],
            'history':history,'seconds_by_stage':times,'optimization_seconds':optimization_seconds,'total_seconds':perf_counter()-started}
    np.savez_compressed(out/'mesh.npz',vertices=p,tetrahedra=t,boundary_faces=f)
    (out/'metrics.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    vtk_fields(out/'field.vtk',p,t,{'displacement_m':fields['displacement']},{'von_mises_Pa':fields['von_mises']})
    print(json.dumps({'free_dofs':met['free_dofs'],'after':after,'mean_loaded_ux_mm':met['mean_loaded_ux_m']*1000,
                      'loss':measured_loss,'optimization_seconds':optimization_seconds,'posthoc_error_percent':record['posthoc_error_percent']},indent=2),flush=True)
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=ROOT/'output/u-goal-adapt/p1-polished/mesh.npz')
    parser.add_argument('--out',type=Path,default=ROOT/'output/speed-quality/quality-full')
    parser.add_argument('--loss-budget',type=float,default=.003)
    parser.add_argument('--fast',action='store_true',help='Apply only fixed-connectivity geometric relaxation')
    args=parser.parse_args();run(args.input.resolve(),args.out.resolve(),args.loss_budget,fast=args.fast)
