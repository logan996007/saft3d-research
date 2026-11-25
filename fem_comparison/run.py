"""Run the same P1 solvers on saved advancing-front and independent TetGen meshes."""
from pathlib import Path
import hashlib
import json
import os
import platform
import sys

# Keep sparse and small dense kernels from competing for a workstation's threads.
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')

import numpy as np
import scipy
import skfem
from . import elasticity, thermal
from .reference import refine_once, independent_displacement, quadratic_heat_check
from saft3d.io import write_json, write_vtk
from saft3d.validation import validate_mesh

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output'/'fem-comparison'
SAFT={'u_saft':'general-u-final','s_saft':'general-slender-final','s_saft_quality':'general-slender-quality'}


def load_mesh(name):
    if name == 'u_cpu_same_nodes':
        # Hold every coordinate, boundary facet and DOF fixed. Change only
        # connectivity; the separate u_coarse control also chooses new points.
        import tetgen
        (vertices, original_cells, boundary), source = load_mesh('u_saft')
        active = np.unique(original_cells)
        vertices = vertices[active]
        boundary = np.searchsorted(active, boundary)
        generator = tetgen.TetGen(vertices, boundary.astype(np.int32),
                                  np.arange(1,len(boundary)+1,dtype=np.int32))
        points, cells, _, _ = generator.tetrahedralize(switches='pYYJMQzO0S0')
        cells = np.asarray(cells,dtype=int).copy()
        assert np.array_equal(points,vertices) and len(np.unique(cells))==len(points)
        xyz = points[cells]
        negative = np.linalg.det(xyz[:,1:]-xyz[:,:1])<0
        cells[negative] = cells[negative][:,[1,0,2,3]]
        checked = validate_mesh(points,cells,boundary)
        if not checked['valid']: raise ValueError(checked['errors'])
        return (points,cells,boundary),dict(source, method='CPU TetGen, frozen SAFT coordinates and boundary triangles')
    if name in SAFT:
        path=ROOT/'output'/SAFT[name]/'mesh.json'
        raw=json.loads(path.read_text(encoding='utf8'))
        arrays=tuple(np.array(raw[k]) for k in ('vertices','tetrahedra','boundary_faces'))
        assert json.loads(path.with_name('report.json').read_text(encoding='utf8'))['validation']['valid']
    else:
        path=OUT/'cpu'/f'{name}.npz'
        with np.load(path,allow_pickle=False) as raw:
            arrays=tuple(raw[k] for k in ('vertices','tetrahedra','boundary_faces'))
    return arrays,{'file':path.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def vtk_fields(path, v, t, point_data, cell_data):
    write_vtk(path,v,t,title=f'P1 finite element solution: {path.stem}')
    with path.open('a',encoding='ascii') as stream:
        for label, n, data in [('POINT_DATA',len(v),point_data),('CELL_DATA',len(t),cell_data)]:
            stream.write(f'{label} {n}\n')
            for name,values in data.items():
                arr=np.asarray(values).reshape(n,-1)
                header=(f'VECTORS {name} double\n' if arr.shape[1]==3 else
                        f'SCALARS {name} double 1\nLOOKUP_TABLE default\n')
                stream.write(header)
                np.savetxt(stream,arr,fmt='%.12g')


def run_all():
    OUT.mkdir(exist_ok=True,parents=True)
    field_dir=OUT/'fields';field_dir.mkdir(exist_ok=True)
    report={'python':sys.version.split()[0],'platform':platform.platform(), 'numpy':np.__version__,
            'scipy':scipy.__version__,'scikit_fem':skfem.__version__,'meshes':{},'independent_elasticity':{}}
    cases=['u_saft','u_cpu_same_nodes','u_coarse','u_native_coarse','u_fine','u_finer',
           's_saft','s_saft_quality','s_coarse','s_quality_coarse','s_native_coarse','s_fine','s_finer']
    for name in cases:
        arrays,source=load_mesh(name)
        # Successively refine the original mesh to distinguish resolution from solvability.
        levels = range(3) if name in ('u_saft','s_saft') else range(1)
        for level in levels:
            if level: arrays=refine_once(*arrays[:2])
            v,t,f=arrays
            label=f'{name}_ref{level}' if level else name
            q,fields=(elasticity if name.startswith('u_') else thermal).solve(v,t,f)
            q.update(source=source,uniform_refinements=level,tetrahedra=len(t),used_nodes=len(np.unique(t)))
            coords=v[t];vol=np.linalg.det(coords[:,1:]-coords[:,:1])/6
            assert vol.min()>0
            edge_sum=sum(np.sum((coords[:,a]-coords[:,b])**2,axis=1) for a in range(4) for b in range(a+1,4))
            mean_ratio=12*(3*vol)**(2/3)/edge_sum
            q['min_mean_ratio']=float(mean_ratio.min())
            q['volume_m3']=float(vol.sum())
            report['meshes'][label]=q
            if 'displacement' in fields:
                vtk_fields(field_dir/f'{label}.vtk',v,t,{'displacement_m':fields['displacement']},
                           {'von_mises_Pa':fields['von_mises']})
            else:
                vtk_fields(field_dir/f'{label}.vtk',v,t,{'temperature_K':fields['temperature']},
                           {'heat_flux_W_m2':fields['heat_flux']})
            print(label,len(t),q['free_dofs'],flush=True)
    for name,order in [('u_saft',1),('u_saft',2),('u_coarse',2),('u_fine',2),('u_finer',2)]:
        (v,t,_),_=load_mesh(name)
        ref=independent_displacement(v,t,order)
        report['independent_elasticity'][f'{name}_p{order}']=ref
        print(f'{name}_p{order}',ref,flush=True)
    implemented=report['meshes']['u_saft']['mean_loaded_ux_m']
    independent=report['independent_elasticity']['u_saft_p1']['mean_loaded_ux_m']
    if not np.isclose(implemented,independent,rtol=1e-8,atol=1e-14):
        raise RuntimeError('Independent P1 elasticity assemblies disagree')
    report['quadratic_heat_check']={}
    for name in ('s_saft','s_coarse'):
        (v,t,_),_=load_mesh(name)
        report['quadratic_heat_check'][name]=quadratic_heat_check(v,t)
    write_json(OUT/'results.json',report)


if __name__=='__main__': run_all()
