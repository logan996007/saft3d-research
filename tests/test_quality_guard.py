import numpy as np
import pytest
from saft3d.quality_guard import measures,quality_cost,shape_data
from saft3d.general_surfaces import make_general_surface
from fem_comparison.audit_goal_u import check_physical_boundary


def test_regular_tetrahedron_metrics_and_orientation():
    v=np.array([[0,0,0],[1,0,0],[.5,np.sqrt(3)/2,0],[.5,np.sqrt(3)/6,np.sqrt(2/3)]])
    t=np.array([[0,1,2,3]])
    m=measures(v,t)
    assert m['mean_ratio']['min']==pytest.approx(1.)
    assert m['dihedral_degrees']['min']==pytest.approx(np.degrees(np.arccos(1/3)))
    assert m['dihedral_degrees']['max']==pytest.approx(m['dihedral_degrees']['min'])
    assert measures(v,t[:,[1,0,2,3]])['min_signed_volume']<0


def test_quality_penalty_gradient_matches_central_difference():
    v=np.array([[.1,.1,.1],[1.,.2,0.],[.3,.95,0.],[.4,.5,.17]])
    t=np.array([[0,1,2,3]]); det=shape_data(v,t)[1]
    loss,grad=quality_cost(v,t,det,.95);assert loss>0
    direction=np.random.default_rng(713).normal(size=v.shape);step=1.e-7
    fd=(quality_cost(v+step*direction,t,det,.95)[0]-quality_cost(v-step*direction,t,det,.95)[0])/(2*step)
    assert (grad*direction).sum()==pytest.approx(fd,rel=2e-7)


def test_rectangular_patch_audit_detects_modified_u():
    p,f,_=make_general_surface('u_prism')
    assert check_physical_boundary(p,f)
    with pytest.raises(ValueError,match='boundary'):
        check_physical_boundary(p,f[:-1])


def test_split_midpoint_relaxation_is_lower_bound_on_recomputed_compliance():
    from itertools import permutations
    from saft3d.quality_guard import _split_candidate
    from fem_comparison.cpu_meshes import extract_boundary
    from fem_comparison.elasticity import solve
    vertices={};cells=[]
    for x in range(3):
        for y in range(3):
            if x==1 and y>=1:continue
            origin=np.array([x,y,0])
            for perm in permutations(range(3)):
                corner=origin.copy();ids=[]
                for step in (None,*perm):
                    if step is not None:corner[step]+=1
                    coord=tuple(corner);ids.append(vertices.setdefault(coord,len(vertices)))
                cells.append(ids)
    p=np.asarray(list(vertices),float);p[:,2]-=.5;t=np.array(cells)
    neg=np.linalg.det(p[t[:,1:]]-p[t[:,:1]])<0
    t[neg]=t[neg][:,[1,0,2,3]]
    f=extract_boundary(t)
    metrics,fields=solve(p,t,f)
    edge=next(tuple(sorted((a,b))) for a,b in zip(t[:,0],t[:,1])
              if np.linalg.norm(p[a]-p[b])>.1 and not np.isclose(np.mean(p[[a,b],1]),3.))
    ids=np.flatnonzero((t==edge[0]).any(1)&(t==edge[1]).any(1))
    gain,midpoint=_split_candidate(p,t,ids,edge,fields['displacement'],0.)
    pieces=[]
    for end in edge:
        child=t[ids].copy();child[t[ids]==end]=len(p);pieces.append(child)
    cells=np.vstack((np.delete(t,ids,axis=0),*pieces));pos=np.vstack((p,midpoint));bnew=extract_boundary(cells)
    adapted,_=solve(pos,cells,bnew)
    assert gain>=0 and adapted['compliance_J']-metrics['compliance_J']>=gain-1e-8
    assert adapted['free_dofs']==metrics['free_dofs']+3
