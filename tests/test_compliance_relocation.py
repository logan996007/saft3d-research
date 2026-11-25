import numpy as np
from fem_comparison import elasticity
from saft3d.compliance_relocation import gradient,free_coordinates


def test_compliance_derivative_includes_moving_traction_and_internal_nodes():
    # Stored validated U mesh has free surface nodes on the loaded end.
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    with np.load(root/'output/local-refinement/u/level0.npz') as data:
        p,t,f=(data[k] for k in ('vertices','tetrahedra','boundary_faces'))
    movable,loaded=free_coordinates(p,f)
    rng=np.random.default_rng(212)
    direction=rng.normal(size=p.shape)*movable
    direction/=np.linalg.norm(direction)
    base,fields=elasticity.solve(p,t,f)
    g,energy,_=gradient(p,t,loaded,fields['displacement'])
    step=1.e-5
    left=elasticity.solve(p-step*direction,t,f)[0]['compliance_J']
    right=elasticity.solve(p+step*direction,t,f)[0]['compliance_J']
    np.testing.assert_allclose(np.sum(energy),base['compliance_J'],rtol=1.e-10)
    np.testing.assert_allclose(np.sum(g*direction),(right-left)/(2*step),rtol=3.e-6,atol=2.e-7)
