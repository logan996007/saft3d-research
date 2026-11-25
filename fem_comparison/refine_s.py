"""Generate, validate and solve the thickness-directed S-prism meshes."""
import argparse
from pathlib import Path

import numpy as np

from fem_comparison.run import ROOT,vtk_fields
from fem_comparison.thermal import solve
from saft3d.general_surfaces import make_general_surface
from saft3d.io import write_json
from saft3d.thickness_front import grow_thickness_front
from saft3d.validation import validate_mesh


def run(backend='cuda',directory=None):
    out=Path(directory) if directory else ROOT/'output'/'local-refinement'/'s'
    out.mkdir(parents=True,exist_ok=True)
    original,faces,info=make_general_surface('slender_s')
    records=[]
    cases={f'level{i}':np.linspace(0.,1.,n+1) for i,n in enumerate((5,9,17))}
    # Same number of free temperatures as level0; smaller steps near the caps.
    cases['wall_graded']=np.array([0.,.15,.38,.62,.85,1.])
    for label,fractions in cases.items():
        # Constant thickness curvature of the benchmark justifies a constant
        # step along z. No degrees of freedom are spent on in-plane refinement.
        points,cells,boundary,gen=grow_thickness_front(original,faces,fractions,backend=backend)
        check=validate_mesh(points,cells,boundary,check_overlap=True)
        if not check['valid']:
            raise RuntimeError(str(check['errors']))
        if not np.isclose(check['positive_volume_sum'],info['analytic_volume'],rtol=1.e-12):
            raise RuntimeError('S prism differs from its analytical volume')
        metrics,fields=solve(points,cells,boundary)
        if metrics['free_dofs']!=gen['free_thermal_dofs']:
            raise RuntimeError('wrong number of active temperature degrees of freedom')
        path=out/f'{label}.npz'
        np.savez_compressed(path,vertices=points,tetrahedra=cells,boundary_faces=boundary)
        vtk_fields(path.with_suffix('.vtk'),points,cells,{'temperature_K':fields['temperature']},
                   {'heat_flux_W_m2':fields['heat_flux']})
        record={'mesh':path.name,'generator':gen,'validation':check,'fem':metrics}
        write_json(path.with_suffix('.json'),record)
        records.append(record)
        print(f"{path.name}: {len(cells)} tetrahedra, {metrics['free_dofs']} free DOFs, "
              f"mean rise {metrics['mean_rise_K']:.7f} K, "
              f"L2 rise error {100*metrics['relative_l2_temperature_rise']:.5f}%, "
              f"flux error {100*metrics['relative_l2_heat_flux']:.5f}%",flush=True)
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--out',type=Path)
    opts=parser.parse_args(); run(opts.backend,opts.out)
