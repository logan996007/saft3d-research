"""Small-strain, static, isotropic elasticity using displacement P1 tetrahedra.

All coordinates are in metres, nodal forces in newtons, and stresses in pascals.
The comparison case is the U bracket in saft3d.general_surfaces. Element
stiffness and constant face traction are integrated exactly for this element.
"""

from time import perf_counter
import warnings

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve


YOUNG = 200e9
POISSON = 0.3
TRACTION = 1e6


def _material_matrix(young, poisson):
    """Isotropic constitutive matrix for engineering shear strains."""
    shear = young / (2 * (1 + poisson))
    lame = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    material = np.diag([2 * shear] * 3 + [shear] * 3)
    material[:3, :3] += lame
    return material


def _strain_displacement(points, cells):
    corners = points[cells]
    # Row j of the Jacobian is the physical edge x_(j+1)-x_0. Thus
    # grad N_(j+1) is column j of its inverse.
    jacobian = corners[:, 1:] - corners[:, :1]
    volumes = np.linalg.det(jacobian) / 6
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("Elasticity requires positive signed tetrahedron volumes")
    gradients = np.empty((len(cells), 4, 3))
    gradients[:, 1:] = np.linalg.inv(jacobian).transpose(0, 2, 1)
    gradients[:, 0] = -gradients[:, 1:].sum(axis=1)
    bmat = np.zeros((len(cells), 6, 12))
    gx, gy, gz = gradients.transpose(2, 0, 1)
    bmat[:, 0, 0::3] = gx
    bmat[:, 1, 1::3] = gy
    bmat[:, 2, 2::3] = gz
    bmat[:, 3, 0::3], bmat[:, 3, 1::3] = gy, gx
    bmat[:, 4, 1::3], bmat[:, 4, 2::3] = gz, gy
    bmat[:, 5, 0::3], bmat[:, 5, 2::3] = gz, gx
    return volumes, bmat


def _surface_weights(points, faces):
    """Nodal integrals of the shape functions over the supplied surface."""
    triangles = points[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    weights = np.zeros(len(points))
    for local_node in range(3):
        np.add.at(weights, faces[:, local_node], areas / 3)
    return weights


def _solve_static(points, cells, fixed, nodal_forces):
    volumes, bmat = _strain_displacement(points, cells)
    material = _material_matrix(YOUNG, POISSON)
    dofs = (3 * cells[:, :, None] + np.arange(3)).reshape(-1, 12)
    element_stiffness = volumes[:, None, None] * (bmat.transpose(0, 2, 1) @ material @ bmat)
    row = np.repeat(dofs, 12, axis=1).ravel()
    col = np.tile(dofs, (1, 12)).ravel()
    stiffness = coo_matrix((element_stiffness.ravel(), (row, col)), shape=(points.size, points.size)).tocsr()
    used = np.zeros(len(points), dtype=bool)
    used[cells.ravel()] = True
    free = (used[:, None] & ~fixed).ravel()
    if np.any(nodal_forces[~used]):
        raise ValueError("Nodal load applied to an unreferenced vertex")
    forces = nodal_forces.ravel()
    u = np.zeros_like(forces, dtype=float)
    if free.any():
        with warnings.catch_warnings():
            warnings.simplefilter('error', MatrixRankWarning)
            u[free] = spsolve(stiffness[free][:, free].tocsc(), forces[free])
    if not np.isfinite(u).all():
        raise ValueError("Nonfinite displacement in elasticity solve")
    nodal_residual = (stiffness @ u - forces).reshape(-1, 3)
    strain = np.einsum('tij,tj->ti', bmat, u[dofs])
    stress = strain @ material.T
    sx, sy, sz, sxy, syz, sxz = stress.T
    von_mises = np.sqrt(0.5*((sx-sy)**2 + (sy-sz)**2 + (sz-sx)**2) + 3*(sxy**2 + syz**2 + sxz**2))
    strain_energy = float(0.5 * np.sum(volumes * np.einsum('ti,ti->t', stress, strain)))
    compliance = float(forces @ u)
    reactions = np.where(fixed, nodal_residual, 0.0)
    external_force = nodal_forces.sum(axis=0)
    force_scale = np.linalg.norm(nodal_forces, axis=1).sum()
    metrics = {
        'young_modulus_Pa': YOUNG, 'poisson_ratio': POISSON,
        'active_vertices': int(used.sum()), 'free_dofs': int(free.sum()),
        'tetrahedra': len(cells), 'reaction_N': reactions.sum(axis=0).tolist(),
        'external_force_N': external_force.tolist(),
        'force_balance_relative': float(np.linalg.norm(reactions.sum(axis=0) + external_force) / max(force_scale, 1e-30)),
        'free_residual_relative': float(np.linalg.norm(nodal_residual.ravel()[free]) / max(np.linalg.norm(forces), 1e-30)),
        'strain_energy_J': strain_energy, 'compliance_J': compliance,
        'energy_identity_relative': float(abs(compliance - 2*strain_energy) / max(abs(compliance), 1e-30)),
        'max_nodal_displacement_m': float(np.linalg.norm(u.reshape(-1, 3)[used], axis=1).max()),
        'max_element_von_mises_Pa': float(von_mises.max()),
    }
    return metrics, {'displacement': u.reshape(-1, 3), 'von_mises': von_mises}


def solve(vertices, tetrahedra, boundary_faces):
    """Clamp the U bracket's left end; apply uniform +x traction on its right end.

    Both end faces lie on y=3, each with area 1 m². The whole left end is
    fixed in x/y/z. The remaining boundary carries zero traction. Returns
    the mean x displacement of the loaded face, integrated by area, plus
    equilibrium diagnostics and per-node/per-element solution fields.
    """
    start = perf_counter()
    points = np.asarray(vertices, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    faces = np.asarray(boundary_faces, dtype=np.int64)
    triangles = points[faces]
    tol = 1e-9
    end = (np.abs(triangles[:, :, 1] - 3.) < tol).all(axis=1)
    clamped_weights = _surface_weights(points, faces[end & (triangles[:, :, 0] <= 1 + tol).all(axis=1)])
    loaded_weights = _surface_weights(points, faces[end & (triangles[:, :, 0] >= 2 - tol).all(axis=1)])
    clamped_area = float(clamped_weights.sum())
    loaded_area = float(loaded_weights.sum())
    if not np.allclose([clamped_area, loaded_area], 1., rtol=1e-9, atol=1e-10):
        raise ValueError(f"U-prism end areas must both be 1 m² (clamped={clamped_area}, loaded={loaded_area})")
    fixed = np.broadcast_to((clamped_weights > 0)[:, None], points.shape)
    nodal_forces = np.zeros_like(points)
    nodal_forces[:, 0] = TRACTION * loaded_weights
    metrics, fields = _solve_static(points, cells, fixed, nodal_forces)
    metrics.update({
        'traction_x_Pa': TRACTION, 'loaded_area_m2': loaded_area, 'clamped_area_m2': clamped_area,
        'mean_loaded_ux_m': float(loaded_weights @ fields['displacement'][:, 0] / loaded_area),
        'solve_seconds': perf_counter() - start,
    })
    return metrics, fields
