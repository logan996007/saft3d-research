from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .surfaces import make_surface
from .mesher import MesherConfig,grow_mesh
from .general_mesher import AdvancingFrontConfig, StalledFront, grow_general_mesh
from .validation import validate_mesh
from .io import read_surface,write_vtk,write_msh,write_json


def main():
    parser=argparse.ArgumentParser(description="GPU-assisted 3D advancing-front meshing with explicit residual-cavity recovery")
    parser.add_argument("--shape",choices=["cube","sphere","dented","tetrahedron","torus","u_prism","hollow_cube","slender_s"],default="cube")
    parser.add_argument("--method",choices=["star","local","bending","thickness"],default="local")
    parser.add_argument("--free-dofs",type=int,default=156,
                        help="target unconstrained P1 displacement DOFs for the benchmark U bracket")
    parser.add_argument("--layers",type=int,default=5,
                        help="number of through-thickness intervals for matching z-extruded caps")
    parser.add_argument("--input",type=Path,help="outward triangular surface: JSON, OBJ or STL")
    parser.add_argument("--backend",choices=["cpu","cuda","auto"],default="cpu")
    parser.add_argument("--size",type=float,default=.6)
    parser.add_argument("--rounds",type=int,default=2,help="maximum advancing-front subdivision depth before kernel closure")
    parser.add_argument("--batch",type=int,default=64)
    parser.add_argument("--subdivisions",type=int,default=1)
    parser.add_argument("--max-new-vertices",type=int,default=256)
    parser.add_argument("--candidate-vertices",type=int,default=24)
    parser.add_argument("--min-aft-quality", type=float, default=AdvancingFrontConfig.min_quality,
                        help="minimum mean ratio for local AFT candidates reusing a vertex")
    parser.add_argument("--min-new-quality", type=float, default=AdvancingFrontConfig.new_vertex_min_quality,
                        help="minimum mean ratio for local AFT candidates adding a vertex")
    parser.add_argument("--optimize",type=int,default=0,help="fixed-boundary quality optimization sweeps")
    parser.add_argument("--completion",choices=['none','tetgen'],default='tetgen',help="explicit CPU residual-cavity completion for local AFT")
    parser.add_argument("--flip-passes",type=int,default=0,help="fixed-boundary 2/3 and 3/2 quality-improving flip passes")
    parser.add_argument("--out",type=Path,default=Path("output/demo"))
    parser.add_argument("--no-viewer",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    try:
        if args.input:
            vertices,faces=read_surface(args.input)
        elif args.shape in ('u_prism','hollow_cube','slender_s'):
            from .general_surfaces import make_general_surface
            vertices,faces,_=make_general_surface(args.shape)
        else:
            vertices,faces=make_surface(args.shape,args.subdivisions)
        if args.method=='local':
            cfg=AdvancingFrontConfig(backend=args.backend,batch_size=args.batch,size=args.size,
                                    max_new_vertices=args.max_new_vertices,candidate_vertices=args.candidate_vertices,
                                    min_quality=args.min_aft_quality,new_vertex_min_quality=args.min_new_quality,
                                    completion=args.completion)
            try:
                out_vertices,tets,run=grow_general_mesh(vertices,faces,cfg)
            except StalledFront as exc:
                write_json(args.out/'partial.json',{'vertices':exc.vertices.tolist(),'tetrahedra':exc.tetrahedra.tolist(),
                            'remaining_front':exc.front.tolist(),'report':exc.report,'complete':False})
                raise
        elif args.method=='bending':
            if args.input or args.shape!='u_prism':
                raise ValueError('the bending sizing profile is calibrated for --shape u_prism')
            from .bending_refinement import grow_bending_front
            out_vertices,tets,faces,run=grow_bending_front(vertices,faces,args.free_dofs,backend=args.backend)
        elif args.method=='thickness':
            if args.layers<1:
                raise ValueError('--layers must be positive')
            from .thickness_front import grow_thickness_front
            fractions=np.linspace(0.,1.,args.layers+1)
            out_vertices,tets,faces,run=grow_thickness_front(vertices,faces,fractions,backend=args.backend,mode='aft')
        else:
            cfg=MesherConfig(backend=args.backend,batch_size=args.batch,growth_rounds=args.rounds,size=args.size)
            out_vertices,tets,run=grow_mesh(vertices,faces,cfg)
        run['generation_tet_count']=len(tets)
        if args.optimize:
            from .optimization import optimize_mesh
            out_vertices,optimization=optimize_mesh(out_vertices,tets,faces,args.optimize,args.backend)
            run['optimization']=optimization
        if args.flip_passes:
            from .topology_optimization import improve_topology
            tets,topology=improve_topology(out_vertices,tets,faces,args.flip_passes)
            run['topology_optimization']=topology
        run['final_tet_count']=len(tets)
        check=validate_mesh(out_vertices,tets,faces,check_overlap=True)
        report={"source":str(args.input) if args.input else args.shape,"run":run,"validation":check}
        write_json(args.out/"report.json",report)
        write_json(args.out/"mesh.json",{"vertices":out_vertices.tolist(),"tetrahedra":tets.tolist(),"boundary_faces":faces.tolist()})
        if not check["valid"]:
            print(json.dumps({"status":"FAILED_VALIDATION","report":str(args.out/"report.json"),"errors":check["errors"]},ensure_ascii=False))
            return 2
        write_vtk(args.out/"mesh.vtk",out_vertices,tets)
        write_msh(args.out/"mesh.msh",out_vertices,tets,faces)
        if not args.no_viewer:
            from .viewer import write_viewer
            write_viewer(args.out/"viewer.html",out_vertices,tets,faces,report)
        print(json.dumps({"status":"VALIDATED","shape":report["source"],"backend":run["backend"],
                         "vertices":len(out_vertices),"tetrahedra":len(tets),
                         "growth_tets":run["growth_tets"],"closure_tets":run["closure_tets"],
                         "meshing_seconds":run["meshing_seconds"],"report":str(args.out/"report.json")},ensure_ascii=False))
        return 0
    except Exception as exc:
        failure={"status":"FAILED","error_type":type(exc).__name__,"error":str(exc)}
        write_json(args.out/"failure.json",failure)
        # A reused output directory must not retain an old successful report
        # as the apparent status of this failed attempt.
        write_json(args.out/"report.json",failure)
        print(json.dumps(failure,ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=="__main__": sys.exit(main())
