import numpy as np
from fem_comparison.elasticity import _material_matrix, _strain_displacement, YOUNG, POISSON, TRACTION
from saft3d.energy_adapt import adaptive_fit, edge_indicators
from saft3d.bending_refinement import fixed_end
from saft3d.general_surfaces import make_general_surface
from saft3d.thickness_front import grow_thickness_front


def test_predicted_edge_gain_equals_explicit_split_residual_and_stiffness():
    # The y=3 face carries traction, so the external part of R_e matters.
    x=np.array([[2.,3,0],[3.,3,0],[2.,3,1],[2.,2,0]])
    t=np.array([[0,1,2,3]])
    surface=np.array([[0,1,2],[0,2,3],[2,1,3],[0,3,1]])
    u=np.array([[1.,3.,2.],[5.,0,1.],[3.,-1.,0.],[4.,1.,2.]])*1e-5
    edges,gains,_,_=edge_indicators(x,t,surface,u)
    material=_material_matrix(YOUNG,POISSON)
    for (a,b),pred in zip(edges,gains):
        x1=np.vstack([x,.5*(x[a]+x[b])]);u1=np.vstack([u,.5*(u[a]+u[b])])
        split=np.vstack([t,t])
        split[0,split[0]==a]=4;split[1,split[1]==b]=4
        vol, shape=_strain_displacement(x1,split)
        ke=vol[:,None,None]*(shape.transpose(0,2,1)@material@shape)
        r=np.array([TRACTION/6 if a<3 and b<3 else 0.,0.,0.])
        k=np.zeros((3,3))
        for cell,stiff in zip(split,ke):
            mid=int(np.flatnonzero(cell==4)[0]); sl=slice(3*mid,3*mid+3)
            r-=stiff[sl,:]@u1[cell].ravel();k+=stiff[sl,sl]
        np.testing.assert_allclose(pred,r@np.linalg.solve(k,r),rtol=1e-12)


def test_disjoint_batch_lower_bound_and_fixed_dof_accounting():
    v,f,_=make_general_surface('u_prism')
    p,t,b,_=grow_thickness_front(v,f,np.array([0.,1.]),backend='cpu',mode='sweep')
    budget=3*(int((~fixed_end(p)).sum())+10)
    p,t,b,history=adaptive_fit(p,t,b,budget)
    assert history[-1]['free_dofs']==budget
    assert all(row2['compliance_J']>=row['compliance_J']+row['predicted_gain_J']-1e-7
               for row,row2 in zip(history[:-1],history[1:]))
    xyz=p[t];assert np.linalg.det(xyz[:,1:]-xyz[:,:1]).min()>0.
