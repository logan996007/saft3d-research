"""Audit U-bracket geometry and re-solve the saved P1 mesh in two assemblies."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from saft3d.validation import validate_mesh
from saft3d.general_surfaces import make_general_surface
from .reference import independent_displacement
from .elasticity import solve
from .run import ROOT, vtk_fields


def _clip(poly,axis,cut,sign):
    result=[]
    for a,b in zip(poly,poly[1:]+poly[:1]):
        da,db=(a[axis]-cut)*sign,(b[axis]-cut)*sign
        if da>=0:result.append(a)
        if (da>0 and db<0) or (da<0 and db>0):result.append(a+da/(da-db)*(b-a))
    return result


def check_physical_boundary(p,f):
    base,_,info=make_general_surface('u_prism')
    polygon=base[:info['polygon_vertices'],:2]
    tol=2.e-10
    if np.any(p<base.min(axis=0)-tol) or np.any(p>base.max(axis=0)+tol):raise ValueError('outside box')
    tri=p[f]; n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])*.5
    assigned=np.zeros(len(f),int); deviations=[]
    for z in (-.5,.5):
        ids=(np.abs(tri[:,:,2]-z)<tol).all(axis=1)&(n[:,2]*z>0)
        assigned+=ids;deviations.append(abs(float(np.abs(n[ids,2]).sum())-7.))
        # No triangle on a cap may bridge the notch between the arms.
        for face in tri[ids,:,:2]:
            cut=list(face)
            for ax,c,s in ((0,1,1),(0,2,-1),(1,1,1)):
                if cut:cut=_clip(cut,ax,c,s)
            if len(cut)>2:
                v=np.asarray(cut)-cut[0]
                if abs(np.cross(v,np.roll(v,-1,axis=0)).sum())*.5>tol:
                    raise ValueError('boundary crosses notch')
    for a,b in zip(polygon,np.roll(polygon,-1,axis=0)):
        edge=b-a; length=np.linalg.norm(edge);tan=edge/length
        normal=np.array([tan[1],-tan[0]])
        delta=tri[:,:,:2]-a
        distance=delta@normal;position=delta@tan
        ids=(np.abs(distance).max(axis=1)<tol)&(position.min(axis=1)>=-tol)&(position.max(axis=1)<=length+tol)&(n[:,:2]@normal>0)
        assigned+=ids;deviations.append(abs(float((n[ids,:2]@normal).sum())-length))
    if not (assigned==1).all() or max(deviations)>tol:raise ValueError('physical boundary mismatch')
    return {'all_faces_on_u_boundary':True,'max_patch_area_error_m2':max(deviations)}


def audit(path,out):
    with np.load(path) as data:
        p,t,f=[data[k] for k in ('vertices','tetrahedra','boundary_faces')]
    domain=check_physical_boundary(p,f)
    check=validate_mesh(p,t,f)
    if not check['valid'] or abs(check['signed_volume_sum']-7)>1.e-9:raise ValueError(check)
    met,fields=solve(p,t,f)
    second=independent_displacement(p,t,1)
    value=met['mean_loaded_ux_m']
    if not np.isclose(second['mean_loaded_ux_m'],value,rtol=1.e-8,atol=0):raise ValueError('second P1 solve disagrees')
    if met['free_dofs']!=3078 or second['free_dofs']!=3078:raise ValueError('budget mismatch')
    old=json.loads((ROOT/'output/fem-comparison/results.json').read_text())['independent_elasticity']['u_finer_p2']['mean_loaded_ux_m']
    new=json.loads((ROOT/'output/u-goal-adapt/reference-sensitivity.json').read_text())['mean_loaded_ux_m']
    report={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'domain':domain,
        'mesh_validation':check,'p1':met,'independent_p1':second,'primary_reference_m':old,
        'sensitivity_reference_m':new,'primary_error_pct':100*abs(value/old-1),
        'sensitivity_error_pct':100*abs(value/new-1)}
    out.mkdir(exist_ok=True,parents=True)
    vtk_fields(out/'field.vtk',p,t,{'displacement_m':fields['displacement']},{'von_mises_Pa':fields['von_mises']})
    (out/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(path,value*1000,report['primary_error_pct'],report['sensitivity_error_pct'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mesh',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args();audit(args.mesh,args.out)
