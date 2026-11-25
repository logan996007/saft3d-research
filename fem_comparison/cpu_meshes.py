"""Generate independent CPU TetGen controls from the original prism surfaces.

Run from the project root with ``python -m fem_comparison.cpu_meshes``.
The node-matched controls keep the original boundary triangles and request a
budget of independently sampled interior points. The refinement controls let
TetGen choose all new points, including points on the same planar boundary.
"""
from itertools import combinations
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc
import tetgen

from saft3d.general_surfaces import make_general_surface


OUT = Path(__file__).resolve().parents[1] / 'output' / 'fem-comparison' / 'cpu'
OUTWARD = np.array([(1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)])
HALTON_SEED = 8731


def prism_interior_points(vertices, n, poly_count):
    """Geometry-only deterministic point set, independent of the FE solution."""
    polygon = vertices[:poly_count, :2]
    sampler = qmc.Halton(d=3, scramble=True, seed=HALTON_SEED)
    result = []
    while len(result) < n:
        points = vertices.min(axis=0) + np.ptp(vertices, axis=0) * sampler.random(256)
        x, y = points[:, 0], points[:, 1]
        inside = np.zeros(len(points), dtype=bool)
        for a,b in zip(polygon, np.roll(polygon, -1, axis=0)):
            if a[1] == b[1]:
                continue
            crosses = (a[1] > y) != (b[1] > y)
            x_cross = a[0] + (b[0]-a[0]) * (y-a[1]) / (b[1]-a[1])
            inside ^= crosses & (x < x_cross)
        result.extend(points[inside])
    return np.array(result[:n])


def extract_boundary(cells):
    oriented = cells[:, OUTWARD].reshape(-1, 3)
    keys = np.sort(oriented, axis=1)
    _, first, inverse, counts = np.unique(keys, axis=0, return_index=True,
                                         return_inverse=True, return_counts=True)
    if counts.max() > 2:
        raise ValueError('nonmanifold tetrahedral face')
    a,b,c = oriented.T
    face_sign = 1 - 2 * (((a>b).astype(int) + (b>c) + (a>c)) % 2)
    signs = np.bincount(inverse.ravel(), weights=face_sign)
    if np.any(signs[counts==2] != 0):
        raise ValueError('internal tetrahedra on same side of common face')
    return oriented[first[counts == 1]]


def audit_boundary(points, boundary, original, faces):
    """Assign every triangle to an oriented input facet and compare each facet's area."""
    triangles = points[boundary]
    crosses = np.cross(triangles[:,1] - triangles[:,0], triangles[:,2] - triangles[:,0])
    area = np.linalg.norm(crosses, axis=1) / 2
    multiplicity = np.zeros(len(boundary),int)
    area_errors = []
    scale = np.ptp(original,axis=0).max()
    for a,b,c in original[faces]:
        normal = np.cross(b-a,c-a)
        area_ref = np.linalg.norm(normal)/2
        normal /= 2*area_ref
        valid = (np.max(np.abs((triangles-a)@normal),axis=1) < 1e-10*scale) & ((crosses@normal)>0)
        for start,end in ((a,b),(b,c),(c,a)):
            inward = np.cross(normal,end-start)
            valid &= np.min((triangles-start)@inward,axis=1) >= -1e-10*scale*np.linalg.norm(inward)
        multiplicity[valid] += 1
        area_errors.append(abs(float(area[valid].sum())/area_ref-1.0))
    if np.any(multiplicity != 1) or max(area_errors)>1e-8:
        raise ValueError(f'prescribed boundary facet mismatch: {np.sum(multiplicity!=1)} faces, {max(area_errors):.6g} relative area error')
    return max(area_errors)


def build_mesh(name, shape, interior_nodes=None, max_volume=None):
    original, faces, meta = make_general_surface(shape)
    if interior_nodes is not None:
        seeds = prism_interior_points(original, interior_nodes, meta['polygon_vertices'])
        inputs = np.vstack((original, seeds))
        options = 'pYYMQz'
    else:
        inputs = original
        options = f'pMq1.8a{max_volume:.9f}Qz'
    markers = np.arange(1, len(faces)+1, dtype=np.int32)
    mesher = tetgen.TetGen(inputs, np.asarray(faces,dtype=np.int32), markers)
    points, cells, _, _ = mesher.tetrahedralize(switches=options)
    points, cells = np.array(points), np.array(cells, dtype=np.int64)
    used, remap = np.unique(cells, return_inverse=True)
    points = points[used]
    cells = remap.reshape(-1,4)
    coords = points[cells]
    d = coords[:,1:]-coords[:,:1]
    volumes = np.einsum('ij,ij->i', d[:,0], np.cross(d[:,1], d[:,2])) / 6
    negative = volumes < 0
    cells[negative,0], cells[negative,1] = cells[negative,1].copy(), cells[negative,0].copy()
    volumes = abs(volumes)
    if volumes.min()<=1e-14 or abs(volumes.sum()/meta['analytic_volume']-1)>1e-10:
        raise ValueError('invalid cell or volume mismatch')
    boundary = extract_boundary(cells)
    boundary_error = audit_boundary(points, boundary, original, faces)
    boundary_ids = np.unique(boundary)
    boundary_keys = {tuple(sorted(map(tuple,f))) for f in points[boundary]}
    exact = boundary_keys == {tuple(sorted(map(tuple,f))) for f in original[faces]}
    if interior_nodes is not None and not exact:
        raise ValueError('frozen coarse control changed prescribed triangles')
    edge_sq = sum(np.sum((coords[:,i]-coords[:,j])**2,axis=1) for i,j in combinations(range(4),2))
    quality = 12*(3*volumes)**(2/3)/edge_sq
    constrained = np.isclose(points[:,1],3)&(points[:,0]<=1+1e-9) if shape=='u_prism' else np.isclose(abs(points[:,2]),meta['extrusion_height']/2)
    nfree=(len(points)-int(constrained.sum()))*(3 if shape=='u_prism' else 1)
    path = OUT / (name+'.npz')
    np.savez_compressed(path, vertices=points, tetrahedra=cells, boundary_faces=boundary)
    return {'file':path.name, 'geometry':shape, 'switches':options, 'input_surface_vertices':len(original),
            'requested_halton_interior_points':interior_nodes, 'halton_seed':HALTON_SEED if interior_nodes else None,
            'max_volume_requested':max_volume, 'vertices':len(points), 'tets':len(cells), 'free_dofs':nfree,
            'boundary_vertices':len(boundary_ids), 'boundary_triangles':len(boundary), 'original_triangles_exact':exact,
            'max_facet_area_relative_error':boundary_error, 'volume':float(volumes.sum()), 'analytic_volume':meta['analytic_volume'],
            'mean_ratio_min':float(quality.min()), 'mean_ratio_median':float(np.median(quality))}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cases={'u_coarse':('u_prism',40,None), 's_coarse':('slender_s',48,None),
           's_quality_coarse':('slender_s',13,None), 'u_fine':('u_prism',None,.003),
           'u_finer':('u_prism',None,.0008), 's_fine':('slender_s',None,.0004),
           's_finer':('slender_s',None,.0001), 'u_native_coarse':('u_prism',None,.08),
           's_native_coarse':('slender_s',None,.0024)}
    records={}
    for name,(shape,n,vol) in cases.items():
        records[name]=build_mesh(name,shape,n,vol)
        r=records[name]; print(f"{name}: {r['vertices']} vertices {r['tets']} tets {r['free_dofs']} free DOFs",flush=True)
    manifest={'generator':'CPU TetGen (independent full-volume generation from original triangular prism PLC)',
              'tetgen_python_version':tetgen.__version__, 'cases':records}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__': main()
