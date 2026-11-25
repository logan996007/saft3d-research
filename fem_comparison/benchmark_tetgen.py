"""Independent CPU TetGen generation benchmark with isolated-process memory peaks.

The U series starts from the same 16 input points / 28 marked triangles at
all sizes. Max. tet volume controls scale; TetGen may refine the boundary.
Timing includes the Python TetGen constructor and materialization of returned
node and element arrays. Disk I/O, quality checks and Python imports are timed
separately from mesh generation. No residual AFT mesh is used.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
import tetgen
from saft3d.general_surfaces import make_general_surface
from saft3d.surfaces import make_surface

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output'/'scale-speed'/'tetgen'
# Based only on volume and cell counts, never on timing or accuracy.
CONFIGS = {'u_1k':('u_prism',.012), 'u_9k':('u_prism',.00140),
           'u_74k':('u_prism',.000164), 'u_590k':('u_prism',.0000207),
           'u_1m':('u_prism',.0000121), 'u_4m7':('u_prism',.00000258),
           'cube_frozen':('cube',None), 'hollow_frozen':('hollow_cube',None)}


def memory():
    if os.name != 'nt':
        import resource
        factor=1 if sys.platform=='darwin' else 1024
        return {'working_set_bytes':None,'peak_working_set_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*factor)}
    class Counters(ctypes.Structure):
        _fields_=[('cb',ctypes.c_uint32),('faults',ctypes.c_uint32)]+[(key,ctypes.c_size_t) for key in ('peak','current','a','b','c','d','page','peakpage')]
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p
    api=ctypes.windll.psapi.GetProcessMemoryInfo
    api.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_uint32]
    if not api(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):raise ctypes.WinError()
    return {'working_set_bytes':int(counters.current),'peak_working_set_bytes':int(counters.peak)}


def tagged_boundary(points, faces, tags, original, prescribed):
    tags=np.array(tags,dtype=int).ravel()-1
    if len(tags)!=len(faces) or tags.min()<0 or tags.max()>=len(prescribed):
        raise ValueError('unexpected boundary facet tags')
    src=original[prescribed]; output=points[faces]
    cross=np.cross(src[:,1]-src[:,0],src[:,2]-src[:,0]); area=np.linalg.norm(cross,axis=1)/2
    normal=cross/(2*area[:,None]); ref=src[tags]; n=normal[tags]
    plane_error=np.max(np.abs(np.einsum('nki,ni->nk',output-ref[:,:1],n)))
    if plane_error>1e-9: raise ValueError('boundary plane mismatch')
    for i in range(3):
        a,b=ref[:,i],ref[:,(i+1)%3]
        side=np.cross(n,b-a);side/=np.linalg.norm(side,axis=1)[:,None]
        if np.einsum('nki,ni->nk',output-a[:,None,:],side).min() < -1e-9:
            raise ValueError('boundary facet extends outside prescribed triangle')
    output_area=np.linalg.norm(np.cross(output[:,1]-output[:,0],output[:,2]-output[:,0]),axis=1)/2
    error=float(np.max(np.abs(np.bincount(tags, weights=output_area,minlength=len(area))/area-1)))
    if error>1e-9: raise ValueError('prescribed facet area mismatch')
    return {'output_boundary_triangles':len(faces),'max_plane_error_m':float(plane_error),'max_relative_tagged_boundary_area_error':error}


def validate(points,cells,volume):
    n=len(cells); qualities=np.empty(n); min_vol=math.inf; vmax=0.;vsum=0.
    for start in range(0,n,40_000):
        block=cells[start:start+40_000]
        if block.min()<0 or block.max()>=len(points):raise ValueError('invalid node ids')
        p=points[block];a,b,c=(p[:,i]-p[:,0] for i in (1,2,3))
        vol=np.einsum('ni,ni->n',a,np.cross(b,c))/6
        if not np.isfinite(vol).all() or vol.min()<=0:raise ValueError('inverted or degenerate tet')
        vmin=float(vol.min());min_vol=min(min_vol,vmin);vmax=max(vmax,float(vol.max()));vsum+=float(vol.sum())
        edge2=sum(np.sum((p[:,i]-p[:,j])**2,axis=1) for i in range(4) for j in range(i+1,4))
        qualities[start:start+len(block)]=12*(3*vol)**(2/3)/edge2
    err=abs(vsum/volume-1.)
    if err>1e-9:raise ValueError('signed volume mismatch')
    return dict(volume=vsum,analytic_volume=volume,relative_volume_error=err,min_tet_volume=min_vol,max_tet_volume=vmax,
                min_mean_ratio=float(qualities.min()),p05_mean_ratio=float(np.quantile(qualities,.05)),median_mean_ratio=float(np.median(qualities)))


def worker(name,trial,out):
    shape,maxvol=CONFIGS[name]
    if shape=='cube':
        p,f=make_surface('cube'); meta={'analytic_volume':8.,'hole_seeds':[]}
    else:p,f,meta=make_general_surface(shape)
    tags=np.arange(1,len(f)+1,dtype=np.int32);tri=np.array(f,np.int32)
    switches=f'pMq1.8a{maxvol:.12f}Qz' if maxvol is not None else 'pYYMQzO0'
    pre=memory();t0=perf_counter();g=tetgen.TetGen(p,tri,tags)
    for seed in meta.get('hole_seeds',[]): g.add_hole(seed)
    prep=perf_counter()-t0;t1=perf_counter();points,cells,_,_=g.tetrahedralize(switches=switches);completed=perf_counter()
    post=memory();duration=completed-t0
    checks=validate(points,cells,meta['analytic_volume'])
    checks.update(tagged_boundary(points,g.trifaces,g.triface_markers,p,f))
    if maxvol is None:
        checks['original_triangle_keys_unchanged']=set(map(tuple,np.sort(g.trifaces,axis=1)))==set(map(tuple,np.sort(f,axis=1)))
        if not checks['original_triangle_keys_unchanged']:raise ValueError('frozen boundary changed')
    audit_time=perf_counter()-completed
    record={'case':name,'trial':trial,'shape':shape,'input_vertices':len(p),'input_triangles':len(f),'output_vertices':len(points),'tetrahedra':len(cells),
            'backend':'CPU TetGen/Python', 'switches':switches,'tetgen_version':tetgen.__version__,'numpy_version':np.__version__,'python':sys.version.split()[0],
            'machine':platform.machine(),'processor':platform.processor(),'platform':platform.platform(),'boundary_refinement':maxvol is not None,
            'radius_edge_quality_parameter':1.8 if maxvol is not None else None,'max_volume_requested':maxvol,
            'prep_seconds':prep,'tetrahedralize_seconds':completed-t1,'generation_seconds':duration,'tet_per_second':len(cells)/duration,
            'process_mem_before':pre,'process_mem_after_generation':post,'audit_seconds':audit_time,'process_mem_after_audit':memory(),'checks':checks}
    out.mkdir(parents=True,exist_ok=True); (out/f'{name}-{trial}.json').write_text(json.dumps(record,indent=2));return record


def summarize_existing(out):
    summary=[]
    for name in CONFIGS:
        paths=sorted(out.glob(f'{name}-[0-9]*.json'))
        if not paths:continue
        records=[json.loads(path.read_text()) for path in paths]
        times=[r['generation_seconds'] for r in records]
        shapes=records[0]['checks']
        summary.append({'case':name,'repeats':len(records),'tetrahedra_range':[min(r['tetrahedra'] for r in records),max(r['tetrahedra'] for r in records)],
                        'generation_seconds_median':statistics.median(times),'seconds_range':[min(times),max(times)],
                        'throughput_median':statistics.median(r['tet_per_second'] for r in records),
                        'peak_process_working_set_bytes':max(r['process_mem_after_generation']['peak_working_set_bytes'] for r in records),
                        'min_mean_ratio':shapes['min_mean_ratio'],'median_mean_ratio':shapes['median_mean_ratio'],
                        'boundary_triangles':shapes['output_boundary_triangles'],'volume':shapes['volume']})
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


def run_series(names,repeats,out,timeout):
    for name in names:
        for trial in range(repeats):
            subprocess.run([sys.executable,'-m','fem_comparison.benchmark_tetgen','--worker',name,'--trial',str(trial),'--out',str(out)],cwd=ROOT,check=True,timeout=timeout)
    summarize_existing(out)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--worker',choices=CONFIGS);parser.add_argument('--trial',type=int,default=0)
    parser.add_argument('--cases',nargs='+',choices=CONFIGS,default=['u_1k','u_9k','u_74k']);parser.add_argument('--repeats',type=int,default=3);parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--timeout',type=float,default=420.,help='wall-clock timeout for each isolated trial')
    parser.add_argument('--summarize',action='store_true',help='rebuild the table from saved completed runs only')
    args=parser.parse_args();out=args.out.resolve()
    if args.summarize:summarize_existing(out)
    elif args.worker:
        row=worker(args.worker,args.trial,out);print(row['case'],row['tetrahedra'],round(row['generation_seconds'],4),flush=True)
    else:
        if args.repeats<1:parser.error('positive repeats required')
        run_series(args.cases,args.repeats,out,args.timeout)
