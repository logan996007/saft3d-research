"""Run midpoint residual h-adaptation using the previous U mesher's parent."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from fem_comparison.run import ROOT, vtk_fields
from fem_comparison.elasticity import solve
from fem_comparison.cpu_meshes import audit_boundary
from saft3d.general_surfaces import make_general_surface
from saft3d.validation import validate_mesh
from saft3d.energy_adapt import adaptive_fit,enriched_strain_metric,metric_fit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=ROOT/'output/local-refinement/u/parent.npz')
    ap.add_argument('--out',type=Path,default=ROOT/'output/u-goal-adapt/energy')
    ap.add_argument('--quality-scale',type=float,default=.25)
    ap.add_argument('--budget',type=int,default=3078)
    ap.add_argument('--validate',action='store_true')
    ap.add_argument('--pilot',type=Path,help='optional separate prior U mesh for enriched H1 metric')
    ap.add_argument('--aspect-limit',type=float,default=5.)
    ap.add_argument('--clamp-nodes',type=int)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    with np.load(args.source) as d:
        mesh=[d[k] for k in ('vertices','tetrahedra','boundary_faces')]
    if args.pilot:
        with np.load(args.pilot) as d:
            metric,est=enriched_strain_metric(d['vertices'],d['tetrahedra'],aspect_limit=args.aspect_limit)
        p,t,f=metric_fit(*mesh,metric,args.budget);history=[dict(est,pilot=str(args.pilot))]
        method='hierarchical P2 strain-gradient metric edge-star bisection'
    else:
        p,t,f,history=adaptive_fit(*mesh,args.budget,quality_scale=args.quality_scale,clamp_nodes=args.clamp_nodes)
        method='independent edge-star residual-driven hierarchical enrichment'
    elapsed=time.perf_counter()-start
    metrics,fields=solve(p,t,f)
    v,tri,_=make_general_surface('u_prism')
    area_error=audit_boundary(p,f,v,tri)
    report={'source':str(args.source),'method':method,
            'history':history,'quality_scale':args.quality_scale,'batch_fraction':.18,
            'target_free_dofs':args.budget,'clamp_node_cap':args.clamp_nodes,
            'max_boundary_facet_area_relative_error':area_error,
            'adaptation_seconds':elapsed,'metrics':metrics}
    np.savez_compressed(args.out/'mesh.npz',vertices=p,tetrahedra=t,boundary_faces=f)
    vtk_fields(args.out/'field.vtk',p,t,{'displacement_m':fields['displacement']},{'von_mises_Pa':fields['von_mises']})
    if args.validate:report['validation']=validate_mesh(p,t,f)
    ref=json.loads((ROOT/'output/fem-comparison/results.json').read_text())['independent_elasticity']['u_finer_p2']['mean_loaded_ux_m']
    report['reference_m']=ref;report['error_fraction']=abs(metrics['mean_loaded_ux_m']/ref-1)
    (args.out/'metrics.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(metrics['free_dofs'],len(t),elapsed,metrics['mean_loaded_ux_m']*1000,report['error_fraction'],flush=True)


if __name__=='__main__':main()
