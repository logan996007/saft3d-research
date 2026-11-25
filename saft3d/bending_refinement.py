"""Geometry-driven local sizing and conforming edge-star bisection.

The default physical size field is defined for the benchmark U bracket in
metres. Both mesh boundaries and tetrahedra are refined by splitting *all*
cells incident on a selected edge. This preserves conformity and each original
planar facet. Mesh generation never consults a displacement reference value.
"""
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, count
import heapq
from time import perf_counter
import numpy as np


def bending_size(xyz, nominal=1.0):
    """Shorter target lengths around the roots and the built-in left end."""
    p = np.asarray(xyz, dtype=float)
    x, y = p[..., 0], p[..., 1]
    root_dist_sq = np.minimum((x-1.0)**2, (x-2.0)**2) + (y-1.0)**2
    clamp_dist_sq = np.maximum(x-1.0, 0.0)**2 + (y-3.0)**2
    return nominal / (1 + 1.10*np.exp(-root_dist_sq/.58**2) +
                      .70*np.exp(-clamp_dist_sq/.55**2))


def fixed_end(points):
    p = np.asarray(points)
    return np.isclose(p[..., 1], 3.0, atol=1e-10, rtol=0) & (p[..., 0] <= 1.0+1e-10)


def constant_size(xyz):
    """Control field for a fixed-budget comparison of spatial weighting."""
    return np.ones(np.asarray(xyz).shape[:-1])


class _Cells:
    """Indexed edge incidences for triangles or tetrahedra, with local updates."""
    def __init__(self, cells):
        self.ids = count()
        self.cells = {}
        self.incident = defaultdict(set)
        for ids in cells:
            self.add(tuple(map(int, ids)))

    def add(self, simplex):
        cell_id = next(self.ids)
        self.cells[cell_id] = simplex
        for edge in combinations(sorted(simplex), 2):
            self.incident[edge].add(cell_id)

    def split(self, edge, midpoint):
        neighbors = set()
        for cell_id in sorted(self.incident.get(edge, ())):
            cell = self.cells.pop(cell_id)
            for adjacent in combinations(sorted(cell), 2):
                bucket = self.incident[adjacent]
                bucket.remove(cell_id)
                if not bucket:
                    del self.incident[adjacent]
            neighbors.update(cell)
            for endpoint in edge:
                self.add(tuple(midpoint if i == endpoint else i for i in cell))
        return neighbors


@dataclass
class Refinement:
    vertices: np.ndarray
    cells: np.ndarray
    boundary: np.ndarray
    details: dict


def refine_bending(vertices, cells, boundary_faces, target_free_vertices, *, size_field=bending_size):
    """Reach an exact *unclamped* vertex budget using longest metric edges.

    Accept 3-node surface triangles for the premeshing stage or positive 4-node
    volume cells. The cells must form a valid complex. Unused nodes are dropped.
    """
    raw = np.asarray(vertices, float)
    simplices = np.asarray(cells, dtype=np.int64)
    exterior = np.asarray(boundary_faces,dtype=np.int64)
    if simplices.ndim != 2 or simplices.shape[1] not in (3,4) or not len(simplices):
        raise ValueError('triangles or tetrahedra required')
    used = np.unique(simplices)
    if not np.isin(exterior,used).all():
        raise ValueError('boundary contains an unused vertex')
    points = list(raw[used])
    volume = _Cells(np.searchsorted(used,simplices))
    surface = _Cells(np.searchsorted(used,exterior))
    initial = len(points)
    free_count = int(np.count_nonzero(~fixed_end(points)))
    if target_free_vertices < free_count:
        raise ValueError('target is smaller than current number of free vertices')
    queue = []
    def enqueue(edge):
        a,b = (points[i] for i in edge)
        # The force bends the arms in the xy plane. A moderate through-thickness
        # factor retains 3D resolution while concentrating nodes in that plane.
        score = np.linalg.norm((b-a)*np.array([1.,1.,.72])) / size_field(.5*(a+b))
        heapq.heappush(queue,(-float(score),edge))
    for edge in volume.incident: enqueue(edge)
    while free_count < target_free_vertices:
        _,edge = heapq.heappop(queue)
        if edge not in volume.incident: continue
        point_id = len(points)
        midpoint = .5*(points[edge[0]]+points[edge[1]])
        points.append(midpoint)
        free_count += int(not fixed_end(midpoint))
        neighbors = volume.split(edge,point_id)
        surface.split(edge,point_id)
        # Every new edge joins the midpoint to a previously existing vertex.
        for neighbor in sorted(neighbors): enqueue((neighbor,point_id))
    p = np.array(points)
    small = bending_size(p) < .67
    return Refinement(p, np.asarray(list(volume.cells.values()),dtype=np.int64),
                      np.asarray(list(surface.cells.values()),dtype=np.int64),
                      {'method':'conforming metric edge-star bisection on CPU','size_field':size_field.__name__,
                       'target_free_vertices':target_free_vertices,'added_vertices':len(p)-initial,
                       'added_in_short_size_zone':int(small[initial:].sum()),
                       'all_in_short_size_zone':int(small.sum()),
                       'zone_definition':'nominal bending_size(x) < 0.67',
                       'free_nodes':free_count,'fixed_nodes':len(p)-free_count})


def build_bending_parent(vertices, faces, backend='cuda'):
    """Subdivide the benchmark surface under the same size field used by AFT."""
    from .general_mesher import AdvancingFrontConfig, grow_general_mesh
    surface = refine_bending(vertices, faces, faces, 32)
    config = AdvancingFrontConfig(backend=backend, size=1., max_new_vertices=3,
                                  candidate_vertices=30, batch_size=32, completion='tetgen')
    p, t, report = grow_general_mesh(surface.vertices, surface.boundary, config,
                                     size_field=bending_size)
    return p, t, surface.boundary, report, surface.details


def bend_levels(parent, free_vertex_budgets, *, size_field=bending_size):
    """Yield the nested, CPU-bisected volume meshes for a completed parent."""
    points, cells, faces, _, _ = parent
    for free_vertices in free_vertex_budgets:
        mesh = refine_bending(points, cells, faces, free_vertices, size_field=size_field)
        yield mesh
        points, cells, faces = mesh.vertices, mesh.cells, mesh.boundary


def grow_bending_front(vertices, faces, free_dofs, *, backend='cuda'):
    """Generate the 3D U benchmark using a localized bending size field.

    The 3-displacement-DOF budget excludes the left clamped end of the 3x3x1 m
    U. AFT and its frozen-interface completion produce the parent mesh. All
    later subdivisions preserve that parent material partition. Reported AFT
    and completion cell counts are counts *before* volume-edge bisection.
    """
    start = perf_counter()
    if not isinstance(free_dofs,(int,np.integer)) or free_dofs%3 or free_dofs<156:
        raise ValueError('U bending profile requires an integer multiple of 3 free DOFs, at least 156')
    parent = build_bending_parent(vertices, faces, backend)
    base = parent[3]
    target = int(free_dofs)//3
    budgets = [b for b in (52,322,1026) if b<target] + [target]
    history = []
    for mesh in bend_levels(parent,budgets):
        history.append(mesh.details)
    report = {
        'algorithm':'SAFT bending-size front and conforming metric edge-star bisection',
        'backend':base['backend'],'device':base['device'],'backend_stats':base['backend_stats'],
        'parent':base,'surface_refinement':parent[4],'volume_refinement':history,
        'free_dofs':int(free_dofs),'remaining_front_faces':base['remaining_front_faces'],
        'growth_tets':base['growth_tets'],'closure_tets':base['closure_tets'],
        'aft_volume_fraction':base.get('aft_volume_fraction',1.),
        'completion_volume_fraction':base.get('completion_volume_fraction',0.),
        'volume_refined_tets':len(mesh.cells),'meshing_seconds':perf_counter()-start,
        'count_scope':'growth/closure are parent cell counts before CPU conforming edge bisection',
    }
    if 'cpu_completion' in base: report['cpu_completion']=base['cpu_completion']
    return mesh.vertices,mesh.cells,mesh.boundary,report
