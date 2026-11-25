"""Independent scikit-fem elasticity assembly and midpoint refinement helpers."""
import numpy as np
from skfem import (MeshTet, Basis, FacetBasis, ElementTetP1, ElementTetP2,
                   ElementVector, LinearForm, asm, condense, solve)
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.models.poisson import laplace, unit_load


def compact_mesh(v, t):
    used, cells = np.unique(t, return_inverse=True)
    return MeshTet(v[used].T.copy(), cells.reshape(t.shape).T.copy())


def refine_once(v, t):
    mesh = compact_mesh(v, t).refined()
    points, cells = mesh.p.T.copy(), mesh.t.T.copy()
    det = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]])
    inv = det < 0
    cells[inv] = cells[inv][:, [1, 0, 2, 3]]
    pattern = [(1,2,3),(0,3,2),(0,1,3),(0,2,1)]
    faces = cells[:,pattern].reshape(-1,3)
    _, idx, counts = np.unique(np.sort(faces,axis=1),axis=0,return_index=True,return_counts=True)
    assert counts.max()==2 and len(cells)==8*len(t)
    return points, cells, faces[idx[counts==1]]


def independent_displacement(v, t, order):
    """Average x-displacement [m] under a uniform 1 MPa end traction."""
    mesh = compact_mesh(v,t)
    left = mesh.facets_satisfying(lambda x: np.isclose(x[1],3.) & (x[0]<1.), boundaries_only=True)
    right = mesh.facets_satisfying(lambda x: np.isclose(x[1],3.) & (x[0]>2.), boundaries_only=True)
    element = ElementVector(ElementTetP1() if order==1 else ElementTetP2())
    basis = Basis(mesh, element, intorder=2)
    boundary = FacetBasis(mesh, element, facets=right, intorder=2)
    stiffness = asm(linear_elasticity(*lame_parameters(200e9,.3)), basis)
    @LinearForm
    def traction(test,w):
        return 1e6*test[0]
    force = asm(traction,boundary)
    fixed = basis.get_dofs(facets=left).all()
    ux = solve(*condense(stiffness, force, D=fixed))
    # Consistent work of the load equals traction times the integral of ux.
    area = float(boundary.dx.sum())
    if not np.isclose(area,1.): raise ValueError('Unexpected loaded area')
    return {'mean_loaded_ux_m':float(force@ux/(1e6*area)), 'order':order,
            'free_dofs':int(basis.N-len(fixed)), 'loaded_area_m2':area}


def quadratic_heat_check(v,t):
    """P2 reproduction check, solving for temperature rise with fourth-order quadrature."""
    mesh = compact_mesh(v,t)
    basis = Basis(mesh, ElementTetP2(), intorder=4)
    bottom, top = mesh.p[2].min(), mesh.p[2].max()
    caps = mesh.facets_satisfying(lambda x: np.isclose(x[2],bottom) | np.isclose(x[2],top), boundaries_only=True)
    fixed = basis.get_dofs(facets=caps).all()
    rise = solve(*condense(10*asm(laplace,basis), 1e4*asm(unit_load,basis), D=fixed))
    z = basis.global_coordinates()[2]
    exact = 500*(z-bottom)*(top-z)
    error = basis.interpolate(rise) - exact
    return {'free_dofs':int(basis.N-len(fixed)), 'relative_l2_temperature_rise':float(
        np.sqrt((error**2*basis.dx).sum()/(exact**2*basis.dx).sum()))}
