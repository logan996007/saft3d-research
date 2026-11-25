"""Linear tetrahedral FEM for an extruded solid heated uniformly from within.

Both end caps in z are held at 300 K. Vertical side faces are insulated.
Coordinates are metres. The exact solution depends only on z, including for
an S-shaped cross-section. The comparison measures errors in temperature
rise and heat flux over the full volume.
"""
from itertools import combinations_with_replacement
from math import factorial, prod

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


CONDUCTIVITY = 10.0       # W/(m K)
SOURCE = 10_000.0         # W/m^3
CAP_TEMPERATURE = 300.0   # K


def _operators(vertices, cells):
    """Return volumes, row-wise shape gradients, global stiffness and load."""
    coordinates = vertices[cells]
    # Rows of edges are the three edge vectors, so inverse(edge_matrix)
    # has the gradients of barycentric coordinates 1, 2, 3 in its columns.
    edges = coordinates[:, 1:] - coordinates[:, :1]
    volumes = np.linalg.det(edges) / 6.0
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0.0):
        raise ValueError("Thermal solve requires finite positive tetrahedron volumes")
    gradients_123 = np.linalg.inv(edges).transpose(0, 2, 1)
    gradients = np.concatenate((-gradients_123.sum(axis=1, keepdims=True), gradients_123), axis=1)
    local_stiffness = CONDUCTIVITY * volumes[:, None, None] * (gradients @ gradients.transpose(0, 2, 1))
    rows = np.repeat(cells, 4, axis=1)
    cols = np.tile(cells, (1, 4))
    stiffness = coo_matrix(
        (local_stiffness.ravel(), (rows.ravel(), cols.ravel())),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    load = np.bincount(cells.ravel(), weights=np.repeat(SOURCE * volumes / 4.0, 4),
                       minlength=len(vertices))
    return volumes, gradients, stiffness, load


_PAIRS = tuple(combinations_with_replacement(range(4), 2))


def _bernstein_mass():
    """Exact volume-average Gram matrix of quadratic Bernstein polynomials."""
    powers = [np.bincount(pair, minlength=4) for pair in _PAIRS]
    weights = [1 if i == j else 2 for i, j in _PAIRS]
    return np.array([
        [wi * wj * factorial(3) * prod(factorial(int(x)) for x in a + b) / factorial(7)
         for wj, b in zip(weights, powers)] for wi, a in zip(weights, powers)
    ])


_MASS2 = _bernstein_mass()


def _error_integrals(local_rise, local_z, gradients, volumes, height):
    """Exact integral of the squared error to the continuous quadratic field."""
    a = SOURCE / (2.0 * CONDUCTIVITY)
    # Products of linear barycentric coordinates span all quadratics. Form
    # coefficients of FE minus exact before squaring to avoid cancellation.
    bernstein = np.column_stack([
        0.5 * (local_rise[:, i] + local_rise[:, j])
        - a * (0.5 * height * (local_z[:, i] + local_z[:, j]) - local_z[:, i] * local_z[:, j])
        for i, j in _PAIRS
    ])
    err2 = volumes @ np.einsum("ei,ij,ej->e", bernstein, _MASS2, bernstein)
    grad_h = np.einsum("ei,eij->ej", local_rise, gradients)
    mean_z = local_z.mean(axis=1)
    # The exact vertical derivative is linear. Its squared integral is its
    # squared mean plus its variance across each tetrahedron.
    var_z = np.sum((local_z - mean_z[:, None])**2, axis=1) / 20.0
    delta_grad = grad_h.copy()
    delta_grad[:, 2] -= a * (height - 2.0 * mean_z)
    grad_err2 = volumes @ (np.sum(delta_grad**2, axis=1) + 4.0 * a**2 * var_z)
    return err2, grad_err2, grad_h


def solve(vertices, tetrahedra, boundary_faces):
    """Solve -div(k grad T)=q for the fixed conductivity, source and cap temperatures.

    Input meshes are validated by the mesher before this call. Boundary face
    indices include every exterior triangle. Unused nodes are omitted from the
    linear system and receive the reference temperature in the output field.
    """
    points = np.asarray(vertices, dtype=float)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    faces = np.asarray(boundary_faces, dtype=np.int64)
    used = np.unique(cells)
    z0, z1 = points[used, 2].min(), points[used, 2].max()
    height = float(z1-z0)
    if height <= 0.0:
        raise ValueError("An extruded domain needs positive height")
    z = points[:, 2] - z0
    tol = 1e-9 * height
    bottom_faces = np.all(np.abs(z[faces]) <= tol, axis=1)
    top_faces = np.all(np.abs(z[faces] - height) <= tol, axis=1)
    bottom, top = np.unique(faces[bottom_faces]), np.unique(faces[top_faces])
    if not len(bottom) or not len(top):
        raise ValueError("Expected flat top and bottom caps on the surface")
    xyz = points[faces]
    normals = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    areas2 = np.linalg.norm(normals, axis=1)
    sides = ~(bottom_faces | top_faces)
    if np.any(np.abs(normals[sides, 2]) > 1e-8 * areas2[sides]):
        raise ValueError("The analytical reference requires vertical side faces")
    fixed = np.union1d(bottom, top)
    free = np.setdiff1d(used, fixed)
    vols, grads, K, F = _operators(points, cells)
    rise = np.zeros(len(points))
    if len(free):
        rise[free] = spsolve(K[free][:, free], F[free])
    if not np.isfinite(rise).all():
        raise RuntimeError("Thermal solution contains nonfinite temperatures")
    local_rise = rise[cells]
    err2, grad_err2, grad_h = _error_integrals(local_rise, z[cells], grads, vols, height)
    volume = float(vols.sum())
    a = SOURCE / (2.0 * CONDUCTIVITY)
    exact_l2_squared = volume * (a * height**2)**2 / 30.0
    exact_grad_squared = volume * (a * height)**2 / 3.0
    outward = F - K @ rise
    cap_out = float(outward[fixed].sum())
    generated = float(SOURCE * volume)
    exact_mean = a * height**2 / 6.0
    exact_peak = a * height**2 / 4.0
    mean_rise = float(vols @ local_rise.mean(axis=1) / volume)
    peak_rise = float(rise[used].max())
    metrics = {
        "tetrahedra": len(cells), "used_nodes": len(used), "free_dofs": len(free),
        "volume_m3": volume, "height_m": height, "conductivity_W_mK": CONDUCTIVITY,
        "source_W_m3": SOURCE, "cap_temperature_K": CAP_TEMPERATURE,
        "mean_rise_K": mean_rise, "exact_mean_rise_K": exact_mean,
        "peak_rise_K": peak_rise, "exact_peak_rise_K": exact_peak,
        "min_rise_K": float(rise[used].min()),
        "relative_mean_rise_error": abs(mean_rise-exact_mean)/exact_mean,
        "relative_peak_rise_error": abs(peak_rise-exact_peak)/exact_peak,
        "relative_l2_temperature_rise": float(np.sqrt(max(0.0, err2)/exact_l2_squared)),
        "relative_l2_heat_flux": float(np.sqrt(grad_err2/exact_grad_squared)),
        "generated_heat_W": generated, "outward_heat_W": cap_out,
        "top_heat_out_W": float(outward[top].sum()), "bottom_heat_out_W": float(outward[bottom].sum()),
        "top_area_m2": float(areas2[top_faces].sum()/2.0),
        "bottom_area_m2": float(areas2[bottom_faces].sum()/2.0),
        "relative_heat_imbalance": abs(generated-cap_out)/generated,
        "free_residual_relative": float(np.linalg.norm(outward[free])/np.linalg.norm(F[used])),
    }
    return metrics, {"temperature": CAP_TEMPERATURE + rise, "heat_flux": -CONDUCTIVITY * grad_h}
