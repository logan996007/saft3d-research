"""Refine the independent numerical reference, separately from all P1 adaptation."""
import json
from pathlib import Path
from time import perf_counter

from saft3d.general_surfaces import make_general_surface
from .refined_cpu import tetrahedralize
from .reference import independent_displacement
from .run import ROOT


def main():
    start = perf_counter()
    v, faces, _ = make_general_surface("u_prism")
    flags = "pMq1.8a0.00045Qz"
    points, tets = tetrahedralize(v, faces, flags)
    result = independent_displacement(points, tets, order=2)
    result.update(tetrahedra=len(tets), mesh_vertices=len(points), tetgen_flags=flags,
                  elapsed_seconds=perf_counter()-start,
                  purpose="Reference sensitivity only; excluded from design and the 3078-DOF comparison.")
    out = ROOT / 'output' / 'u-goal-adapt'
    out.mkdir(exist_ok=True, parents=True)
    (out/'reference-sensitivity.json').write_text(json.dumps(result, indent=2), encoding='utf8')
    print(result, flush=True)


if __name__ == '__main__':
    main()
