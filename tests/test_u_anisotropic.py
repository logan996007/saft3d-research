import numpy as np
from saft3d.u_anisotropic import patch_hessian, adapted_section, extrude_section
from fem_comparison.elasticity import solve


def test_gradient_recovery_reproduces_quadratic_hessian():
    rng=np.random.default_rng(11)
    points=rng.uniform(-1.,1.,(85,3)); volumes=rng.uniform(.01,.04,len(points))
    h=rng.normal(size=(3,3,3));h=.5*(h+h.swapaxes(0,1))
    # gradient[k,c,d]=partial_d u_c(x[k]); Hessian[a,d,c]=partial_a partial_d u_c.
    gradients=np.einsum('ka,adc->kcd',points,h)+rng.normal(size=(1,3,3))
    actual=patch_hessian(points,gradients,volumes)
    np.testing.assert_allclose(actual,np.broadcast_to(h,actual.shape),rtol=1e-11,atol=1e-12)


def test_metric_cvt_preserves_u_and_exact_free_dof_budget():
    rng=np.random.default_rng(17)
    centres=rng.uniform(0.,3.,(250,2)); gram=np.broadcast_to(np.eye(2),(250,2,2)).copy()
    # Prescribe stronger indicator around the re-entrant corner.
    gram[:,0,0]+=30*np.exp(-np.sum((centres-[1,1])**2,axis=1)/.2)
    xy,tri,nb=adapted_section(centres,gram,np.ones(250),90,resolution=28,iterations=3)
    points,cells,bound,_=extrude_section(xy,tri,nb,3,backend='cpu')
    metrics,_=solve(points,cells,bound)
    assert metrics['free_dofs']==810
    corners=points[cells];vol=np.linalg.det(corners[:,1:]-corners[:,:1])/6
    assert np.all(vol>0) and np.isclose(vol.sum(),7.)
    assert np.isclose(metrics['loaded_area_m2'],1.) and np.isclose(metrics['clamped_area_m2'],1.)
