"""Benchmark dynamic batched prism AFT with full host connectivity and certificates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
import numpy as np
from saft3d.backend import create_backend
from saft3d.batched_prism import advance_batched_prisms
from saft3d.validation import validate_mesh
from .cpu_meshes import extract_boundary
from .verify_aft_seed import cap_and_layers,unordered_tet_keys
from .audit_goal_u import check_physical_boundary


ROOT=Path(__file__).resolve().parents[1]


def input_and_reference(seed,intervals):
    """Lift an independently generated six-vertex local-AFT slab, for QA only."""
    with np.load(seed,allow_pickle=False) as d:
        mesh,source_cells,front=d['vertices'],d['tetrahedra'],d['boundary_faces']
    decode=perf_counter()
    cap,original=cap_and_layers(mesh,front)
    decode=perf_counter()-decode
    if intervals<1:raise ValueError('positive layer count required')
    n=original.shape[1]
    if intervals==len(original)-1:return mesh,cap,original,source_cells,decode
    xy=mesh[original[0],:2]
    z=np.linspace(mesh[:,2].min(),mesh[:,2].max(),intervals+1)
    pts=np.column_stack((np.tile(xy,(len(z),1)),np.repeat(z,n)))
    levels=np.arange(len(pts),dtype=np.int64).reshape(len(z),n)
    mapped=np.empty(len(mesh),dtype=np.int64);mapped[original.ravel()]=np.arange(original.size)
    cells=mapped[source_cells]
    slab=cells[np.all(cells<2*n,axis=1)]
    if len(slab)!=3*len(cap):raise ValueError('expected independently generated layer-zero volume cells')
    expected=np.concatenate([slab+i*n for i in range(intervals)])
    return pts,cap,levels,expected,decode


def verify_large(p,t,expected):
    start=perf_counter()
    if not np.array_equal(unordered_tet_keys(t),unordered_tet_keys(expected)):
        raise ValueError('3D cells do not match the independent AFT seed/affine slab lift')
    b=extract_boundary(t)
    domain=check_physical_boundary(p,b)
    volumes=np.linalg.det(p[t[:,1:]]-p[t[:,:1]])/6.
    if volumes.min()<=0 or not np.isclose(volumes.sum(),7.,rtol=1e-12):
        raise ValueError('U mesh positive volume/coverage mismatch')
    return {'same_canonical_tets_as_independent_aft':True,'positive_tets':len(t),
            'oriented_boundary_faces':len(b),'signed_volume':float(volumes.sum()),
            **domain,'check_s':perf_counter()-start}


def worker(seed,out,backend,layers,repeats):
    p,c,l,expected,decode=input_and_reference(seed,layers)
    start=perf_counter()
    engine=create_backend(backend)
    init=perf_counter()-start
    t,info=advance_batched_prisms(p,c,l,engine)
    first=perf_counter()-start
    oracle=verify_large(p,t,expected)
    # Same exact-output oracle for all timed repetitions, including the time
    # to bring CUDA connectivity and failure status back to host.
    trials=[]
    for _ in range(repeats):
        start=perf_counter()
        other,stats=advance_batched_prisms(p,c,l,engine)
        generation=perf_counter()-start
        if not np.array_equal(other,t):raise RuntimeError('repeated mesh differs')
        full=perf_counter()-start
        trials.append({'generation_s':generation,'generation_and_match_s':full,
                       'core_with_transfer_s':stats['timings_seconds']['batch_front']})
    mid=median(r['generation_s'] for r in trials)
    result={'backend':backend,'layers':layers,'prisms':len(c)*layers,'output_tets':len(t),
            'output_vertices':len(p),'source_sha256':hashlib.sha256(seed.read_bytes()).hexdigest(),
            'fresh_process_first_generate_s':first,'seed_cap_decode_s':decode,
            'first_including_seed_cap_decode_s':first+decode,'initialization_s':init,
            'first_generate_after_init_s':info['total_seconds'],'warm_engine_median_s':mid,
            'warm_engine_median_including_repeat_match_s':median(r['generation_and_match_s'] for r in trials),
            'warm_tets_per_s':len(t)/mid,'device':engine.device_info,'repeats':trials,
            'batch':info,'independent_geometry':oracle}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(backend,layers,len(t),f'first={first:.6f}s warm={mid:.6f}s',flush=True)


def bench(seed,out,layers,repeats,baseline):
    out.mkdir(parents=True,exist_ok=True);rows=[];env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]='1'
    with TemporaryDirectory(prefix='prism-aft-nvrtc-') as cache:
        for j,count in enumerate(layers):
            for backend in ('cpu','cuda'):
                # Warm-disk runs always launch in independent processes. Only
                # the optional first CUDA diagnostic uses an isolated empty cache.
                modes=['existing-disk']
                if j==0 and backend=='cuda':modes.insert(0,'empty-disk')
                for mode in modes:
                    child=env.copy()
                    if mode=='empty-disk':child['CUPY_CACHE_DIR']=cache
                    record=out/f'{backend}-{count}-{mode}.json'
                    args=[sys.executable,'-m','fem_comparison.benchmark_batched_prism','--worker',backend,
                          '--seed',str(seed),'--layers',str(count),'--repeats',str(repeats),'--out',str(record)]
                    subprocess.run(args,env=child,cwd=ROOT,timeout=300,check=True)
                    data=json.loads(record.read_text())
                    rows.append({'mode':mode,'record':record.name,**data})
    result={'algorithm':'Batched finite local advancing fronts on independent straight triangular prisms',
            'python':sys.version,'platform':platform.platform(),'numpy':np.__version__,
            'method':'One common embedded cap and prescribed sidewall triangle diagonals. Geometry evaluation of all fifteen 4-of-6 vertex choices per prism, then three dynamic oriented boundary advances. CPU vectorized batches; one CUDA thread per prism. All benchmark returns include full host connectivity, all geometry checks, no CPU closure.',
            'input_scope':'Seed U section; larger cases increase vertical intervals at fixed cap, not an arbitrary unstructured domain.',
            'timing_scope':'Generation includes all in-function validation, CPU mask construction, transfers, per-prism determinants/front/closure/volume checks. Fresh-process first call includes backend creation. Warm medians reuse an engine. Independent oracle after first call is reported separately; timed repetitions match that output exactly.',
            'rows':rows}
    if baseline.is_file():
        old=json.loads(baseline.read_text())
        result['legacy_single_run_s']={b:old[b]['generation_s'] for b in ('cpu','cuda')}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    for row in rows:
        print(row['output_tets'],row['backend'],row['mode'],row['fresh_process_first_generate_s'],row['warm_engine_median_s'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=Path,default=ROOT/'output/u-goal-adapt/p1-anisotropic/actual-aft-seed.npz')
    parser.add_argument('--layers',type=int,nargs='+',default=[2,20,60])
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--out',type=Path,default=ROOT/'output/optimized-batch/prism')
    parser.add_argument('--legacy',type=Path,default=ROOT/'output/scale-speed/u3078-prism/summary.json')
    parser.add_argument('--worker',choices=['cpu','cuda'])
    args=parser.parse_args()
    if args.worker:worker(args.seed.resolve(),args.out.resolve(),args.worker,args.layers[0],args.repeats)
    else:bench(args.seed.resolve(),args.out.resolve(),args.layers,args.repeats,args.legacy)
