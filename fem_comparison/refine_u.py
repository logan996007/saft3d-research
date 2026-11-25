"""Generate and solve locally refined U brackets at fixed free P1-node budgets."""
import argparse
import json
from pathlib import Path
import numpy as np
from saft3d.bending_refinement import bending_size, build_bending_parent, bend_levels, constant_size
from saft3d.general_surfaces import make_general_surface
from saft3d.validation import validate_mesh
from .cpu_meshes import audit_boundary
from .elasticity import solve
from .run import ROOT, vtk_fields


def generate(out, budgets=(52,322,1026), *, control_parent=None):
    out.mkdir(parents=True,exist_ok=True)
    original, facets, _ = make_general_surface('u_prism')
    if control_parent is None:
        points,tets,boundary,parent,plc_details=build_bending_parent(original,facets)
        field=bending_size
    else:
        with np.load(control_parent/'parent.npz') as data:
            points,tets,boundary=(data[k] for k in ('vertices','tetrahedra','boundary_faces'))
        parent=json.loads((control_parent/'parent.json').read_text(encoding='utf8'))
        plc_details=json.loads((control_parent/'level0.json').read_text(encoding='utf8'))['surface_refinement']
        field=constant_size
    np.savez_compressed(out/'parent.npz',vertices=points,tetrahedra=tets,boundary_faces=boundary)
    (out/'parent.json').write_text(json.dumps(parent,indent=2),encoding='utf8')
    reference=json.loads((ROOT/'output/fem-comparison/results.json').read_text(encoding='utf8'))
    reference_m=reference['independent_elasticity']['u_finer_p2']['mean_loaded_ux_m']
    parent_data=points,tets,boundary,parent,plc_details
    for i,(budget,mesh) in enumerate(zip(budgets,bend_levels(parent_data,budgets,size_field=field))):
        points,tets,boundary=mesh.vertices,mesh.cells,mesh.boundary
        label=f'level{i}'
        np.savez_compressed(out/f'{label}.npz',vertices=points,tetrahedra=tets,boundary_faces=boundary)
        metrics,fields=solve(points,tets,boundary)
        assert metrics['free_dofs']==3*budget
        area_error=audit_boundary(points,boundary,original,facets)
        check=validate_mesh(points,tets,boundary)
        if not check['valid']:
            raise RuntimeError(check['errors'])
        vtk_fields(out/f'{label}.vtk',points,tets,{'displacement_m':fields['displacement']},
                   {'von_mises_Pa':fields['von_mises']})
        report={'case':'u_prism','vertices':len(points),'tets':len(tets),'p1_elasticity':metrics,
                'local_volume_weight':field.__name__,
                'reference_m':reference_m,'displacement_error_fraction':1-metrics['mean_loaded_ux_m']/reference_m,
                'surface_refinement':plc_details,'volume_refinement':mesh.details,
                'max_relative_original_boundary_area_error':area_error,
                'generation':{'parent_report':'parent.json','AFT_tets':parent['growth_tets'],
                              'CPU_closure_tets':parent['closure_tets'],
                              'AFT_volume_fraction':parent.get('aft_volume_fraction',1.),
                              'CPU_completion_volume_fraction':parent.get('completion_volume_fraction',0.),
                              'child_cells':'CPU midpoint splits of unchanged parent tetrahedral material regions'},
                'mesh_validation':check}
        (out/f'{label}.json').write_text(json.dumps(report,indent=2),encoding='utf8')
        print(label,len(points),len(tets),metrics['free_dofs'],metrics['mean_loaded_ux_m']*1000,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--uniform-control',action='store_true',help='reuse the actual U parent; turn off position weighting only when bisecting volume-cell edges')
    args=parser.parse_args()
    base=ROOT/'output/local-refinement/u'
    if args.uniform_control:
        generate(args.out or base.with_name('u-uniform-control'),budgets=(52,322),control_parent=base)
    else:
        generate(args.out or base)
