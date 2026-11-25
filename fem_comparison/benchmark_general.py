"""Matched CPU/CUDA wall-time measurements of the general global 3-D AFT."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/'output/scale-speed/general'
CASES = {
    'cube-small':('cube',1,1.0),
    'cube-medium':('cube',6,.8),
    'hollow-large':('hollow_cube',32,.8),
    'sphere-stress':('sphere-128',64,.32),
}


def save(path, obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf8')


def worker(case,backend,out):
    from functools import wraps
    import numpy as np
    from saft3d import general_mesher as gm
    from saft3d.surfaces import cube,sphere
    from saft3d.general_surfaces import make_general_surface
    name,limit,edge=CASES[case]
    v,f=cube() if name=='cube' else sphere(2) if name=='sphere-128' else make_general_surface(name)[:2]
    cfg=gm.AdvancingFrontConfig(backend=backend,max_new_vertices=limit,size=edge,
        batch_size=32,candidate_vertices=24,completion='tetgen')
    stages=defaultdict(float)
    def timed(label,fun):
        @wraps(fun)
        def call(*a,**kw):
            t0=time.perf_counter()
            try: return fun(*a,**kw)
            finally: stages[label]+=time.perf_counter()-t0
        return call
    # This subprocess handles one measured generation; patches are local to it.
    gm._updated_front=timed('front_topology',gm._updated_front)
    gm._front_delta=timed('front_topology',gm._front_delta)
    gm.FrontGeometry.valid_candidate=timed('front_geometric_legality',gm.FrontGeometry.valid_candidate)
    gm.FrontGeometry.__init__=timed('front_snapshot',gm.FrontGeometry.__init__)
    gm.check_surface=timed('input_checks',gm.check_surface)
    gm.check_embedded_surface=timed('input_checks',gm.check_embedded_surface)
    gm.create_backend=timed('backend_initialization',gm.create_backend)
    if case=='sphere-stress':
        real_create=gm.create_backend
        def create_tracked(mode):
            engine=real_create(mode);propose=engine.propose_local;last=0.
            def report_progress(*args,**kwargs):
                nonlocal last
                now=time.perf_counter()
                if now-last>=2:
                    # Snapshot after the previous commit. These count existing
                    # provisional cells, not mere candidates or iterations.
                    caller=sys._getframe(1).f_locals
                    tets=caller['tets'];front=caller['front'];history=caller['history']
                    save(out/'progress.json',{'elapsed_generation_s':now-start,
                         'provisional_aft_tets':len(tets),'residual_front_faces':len(front),
                         'successful_batches':len(history)})
                    last=now
                return propose(*args,**kwargs)
            engine.propose_local=report_progress;return engine
        gm.create_backend=create_tracked
    start=time.perf_counter()
    save(out/'started.json',{'perf_counter':start,'case':case,'backend':backend})
    try:
        p,c,run=gm.grow_general_mesh(v,f,cfg)
    except gm.StalledFront as exc:
        save(out/'run.json',{'case':case,'backend':backend,'status':'stalled',
             'generation_s':time.perf_counter()-start,'partial_aft_tets':len(exc.tetrahedra),
             'remaining_front_faces':len(exc.front),'detail':str(exc),'partial_run':exc.report})
        return
    generation=time.perf_counter()-start
    backend_stats=run['backend_stats']
    for key,duration in backend_stats['timings_seconds'].items():stages['backend_'+key]=duration
    if 'cpu_completion' in run:
        # Includes the final CPU recovery attempt only. Earlier attempts are
        # accounted for in remaining host time if there was backtracking.
        stages['successful_cpu_completion']=run['cpu_completion']['wall_seconds']
    stages['remaining_host_time']=generation-sum(stages.values())
    np.savez_compressed(out/'mesh.npz',vertices=p,tetrahedra=c,boundary_faces=f)
    import dataclasses
    result={'status':'ok','case':case,'shape':name,'backend':backend,
            'input_vertices':len(v),'input_faces':len(f),'config':dataclasses.asdict(cfg),
            'output_vertices':len(p),'output_tets':len(c),'aft_tets':run['growth_tets'],
            'legality_stats':run['legality_stats'],
            'provisional_aft_tets':run.get('provisional_aft_tets',run['growth_tets']),
            'rolled_back_aft_indices':run.get('rolled_back_aft_indices',[]),
            'completion_attempts':run.get('completion_attempts',[]),
            'cpu_completion_tets':run['closure_tets'],'aft_volume_fraction':run.get('aft_volume_fraction',1.0),
            'front_iterations':run['iterations'],'proposals':run['proposals'],
            'device':run['device'],'kernel_calls':backend_stats['kernel_calls'],
            'stage_seconds':dict(stages),'generation_s':generation,'clock':'perf_counter',
            'mesh_sha256':hashlib.sha256((out/'mesh.npz').read_bytes()).hexdigest()}
    save(out/'run.json',result)


def invoke_worker(case,backend,directory,timeout,validate):
    import numpy as np
    from saft3d.validation import validate_mesh
    directory.mkdir(parents=True,exist_ok=True)
    started=directory/'started.json'
    started.unlink(missing_ok=True)
    log=directory/'process.log'
    command=[sys.executable,'-m','fem_comparison.benchmark_general','--worker',case,backend,
             '--out',str(directory)]
    start=time.perf_counter();generation_start=None
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]='1'
    with log.open('w',encoding='utf8') as stream:
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env)
        while process.poll() is None:
            if generation_start is None and started.exists():
                try: generation_start=json.loads(started.read_text())['perf_counter']
                except (ValueError,OSError):pass
            elapsed=time.perf_counter()-(generation_start if generation_start is not None else start)
            if elapsed>timeout:
                process.kill();process.wait()
                record={'case':case,'backend':backend,'status':'timeout','timeout_s':timeout,
                        'generation_started':generation_start is not None,'elapsed_wall_s':time.perf_counter()-start}
                if (directory/'progress.json').exists():
                    record['last_work_snapshot']=json.loads((directory/'progress.json').read_text())
                save(directory/'run.json',record)
                return record
            time.sleep(.1)
    if process.returncode:
        record={'case':case,'backend':backend,'status':'failed','exit_code':process.returncode,
                'detail':log.read_text(encoding='utf8')[-1600:]}
        save(directory/'run.json',record);return record
    record=json.loads((directory/'run.json').read_text())
    record['process_wall_s']=time.perf_counter()-start
    if record['status']!='ok':
        save(directory/'run.json',record);return record
    if validate:
        with np.load(directory/'mesh.npz') as d:
            result=validate_mesh(d['vertices'],d['tetrahedra'],d['boundary_faces'])
        save(directory/'geometry.json',result)
        record['independent_validation_s']=result['validation_seconds']
        record['valid']=result['valid']
        if not result['valid']:raise RuntimeError(result['errors'])
    save(directory/'run.json',record);return record


def compare_meshes(path_a,path_b):
    import numpy as np
    def key_rows(c):
        c=np.sort(c,axis=1)
        return c[np.lexsort(c[:,::-1].T)]
    with np.load(path_a) as a,np.load(path_b) as b:
        pa,pb=a['vertices'],b['vertices']
        return {'same_boundary':np.array_equal(a['boundary_faces'],b['boundary_faces']),
                'equal_connectivity':np.array_equal(key_rows(a['tetrahedra']),key_rows(b['tetrahedra'])),
                'matching_vertices_1e-10':pa.shape==pb.shape and np.allclose(pa,pb,rtol=0.,atol=1.e-10),
                'max_coordinate_difference':float(abs(pa-pb).max()) if pa.shape==pb.shape else None}


def prism_worker(mesh,backend,out):
    import numpy as np
    from fem_comparison.verify_aft_seed import cap_and_layers,unordered_tet_keys,oriented_triangle_keys
    from saft3d.thickness_front import _advance_with_local_aft
    from fem_comparison.cpu_meshes import extract_boundary
    from saft3d.validation import validate_mesh
    from fem_comparison import elasticity
    with np.load(mesh,allow_pickle=False) as d:
        points,faces=d['vertices'],d['boundary_faces']
    start=time.perf_counter()
    cap,layers=cap_and_layers(points,faces)
    tets,record=_advance_with_local_aft(points,cap,layers,backend)
    generation=time.perf_counter()-start
    with np.load(mesh,allow_pickle=False) as d:
        same=np.array_equal(unordered_tet_keys(tets),unordered_tet_keys(d['tetrahedra']))
    same_surface=np.array_equal(oriented_triangle_keys(extract_boundary(tets)),oriented_triangle_keys(faces))
    if not same or not same_surface:raise ValueError('prism generation differed from declared seed')
    # Only the cap subdivision, z planes and compatible wall diagonals were
    # supplied to AFT. The seed tetrahedra were loaded solely for comparison.
    checks=validate_mesh(points,tets,faces)
    if not checks['valid']:raise ValueError(checks['errors'])
    met,_=elasticity.solve(points,tets,faces)
    if met['free_dofs']!=3078:raise ValueError('wrong U displacement budget')
    np.savez_compressed(out/'mesh.npz',vertices=points,tetrahedra=tets,boundary_faces=faces)
    result={'backend':backend,'repeats':1,'source':str(mesh),'source_sha256':hashlib.sha256(mesh.read_bytes()).hexdigest(),
            'cap_vertex_count':len(layers[0]),'cap_triangles':len(cap),'layers':len(layers),'vertices':len(points),
            'tetrahedra':len(tets),'input_surface_triangles':len(faces),'generation_s':generation,
            'tets_per_s':len(tets)/generation,'exact_seed_tetrahedra':bool(same),'exact_seed_oriented_surface':bool(same_surface),
            'all_prism_aft':record,'free_dofs':met['free_dofs'],'displacement_m':met['mean_loaded_ux_m'],
            'validation_s':checks['validation_seconds'],'valid':checks['valid']}
    save(out/'run.json',result);save(out/'geometry.json',checks)
    print(backend,len(tets),generation,flush=True)


def prism_pair(mesh,out,timeout):
    env=os.environ.copy()
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):env[k]='1'
    results={}
    for backend in ('cpu','cuda'):
        directory=out/backend;directory.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-m','fem_comparison.benchmark_general','--prism-worker',backend,
                 '--mesh',str(mesh.resolve()),'--out',str(directory)]
        subprocess.run(command,cwd=ROOT,check=True,timeout=timeout,env=env)
        results[backend]=json.loads((directory/'run.json').read_text())
    results['cpu_over_cuda_single_run_ratio']=results['cpu']['generation_s']/results['cuda']['generation_s']
    results['equivalence']=compare_meshes(out/'cpu/mesh.npz',out/'cuda/mesh.npz')
    save(out/'summary.json',results)


def experiment(out,repeats,timeout,cases):
    from statistics import median
    result={'python':sys.version,'platform':platform.platform(),'cpu':platform.processor(),
            'repeats_requested':repeats,'timeout_per_generation_seconds':timeout,
            'protocol':'Fresh process for each run, existing CUDA disk cache, one BLAS thread. Wall time from grow_general_mesh entry to return includes device initialization and transfers. Readback validation and output serialization excluded. CPU is the NumPy/Python implementation of the same general advancing-front algorithm.',
            'cases':{}}
    for case in cases:
        group={};first_paths={};stop=set()
        # Alternate execution order to limit one-sided timing bias.
        for i in range(repeats):
            for backend in (('cuda','cpu') if i%2==0 else ('cpu','cuda')):
                if backend in stop:continue
                directory=out/case/backend/f'run{i+1}'
                r=invoke_worker(case,backend,directory,timeout,validate=(i==0))
                print(case,backend,i+1,r['status'],r.get('generation_s',r.get('elapsed_wall_s','')),flush=True)
                runs=group.setdefault(backend,{'runs':[]})['runs'];runs.append(r)
                if r['status']!='ok':
                    stop.add(backend);continue
                if i==0:first_paths[backend]=directory/'mesh.npz'
                else:
                    eq=compare_meshes(first_paths[backend],directory/'mesh.npz')
                    if not all(eq[k] for k in ('same_boundary','equal_connectivity','matching_vertices_1e-10')):
                        raise ValueError(f'Repeated {case} {backend} mesh differs: {eq}')
            result['cases'][case]=group; save(out/'summary.json',result)
        for backend in ('cpu','cuda'):
            runs=group[backend]['runs']
            if all(r['status']=='ok' for r in runs):
                med=median(r['generation_s'] for r in runs)
                group[backend].update({'median_generation_s':med,'median_tets_per_s':median(r['output_tets']/r['generation_s'] for r in runs),
                                       'median_stages_s':{k:median(r['stage_seconds'][k] for r in runs) for k in runs[0]['stage_seconds']}})
        if len(first_paths)==2:
            group['cross_backend_comparison']=compare_meshes(first_paths['cpu'],first_paths['cuda'])
        if all('median_generation_s' in group[b] for b in ('cpu','cuda')):
            group['cpu_over_cuda_time_ratio']=group['cpu']['median_generation_s']/group['cuda']['median_generation_s']
        elif group['cpu']['runs'][0]['status']=='timeout' and group['cpu']['runs'][0]['generation_started'] and 'median_generation_s' in group['cuda']:
            group['observed_cpu_timeout_to_cuda_median_lower_bound']=timeout/group['cuda']['median_generation_s']
        save(out/'summary.json',result)


def write_report(out):
    r=json.loads((out/'summary.json').read_text())
    lines=['# 通用三维推进前沿墙钟计时','',
           '从整个输入曲面开始推进。CPU 是同一 Python/NumPy 实现的参考后端，CUDA 在同一台 GTX 1660 Ti 上执行候选、质量和碰撞内核。每次启动独立进程并使用已存在的 CUDA 磁盘缓存，各种 CPU 数值库设置为单线程。下表计时包括输入检查、后端初始化、计算、内存传输、同步和前沿更新，独立验证及文件写出另计。','',
           '| 输入 | 后端 | 边界三角面 | 输出点 / 四面体 | 保留 AFT / CPU 补全单元 | 三次生成中位数 (s) | 单元/s | 首次独立验证 (s) |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for name,case in r['cases'].items():
        for b in ('cpu','cuda'):
            s=case[b]; run=s['runs'][0]
            if 'median_generation_s' not in s:
                lines.append(f"| {name} | {b} | | | | {run['status']} | | |");continue
            lines.append(f"| {name} | {b} | {run['input_faces']} | {run['output_vertices']} / {run['output_tets']} | {run['aft_tets']} / {run['cpu_completion_tets']} | {s['median_generation_s']:.3f} | {s['median_tets_per_s']:.1f} | {run['independent_validation_s']:.3f} |")
    lines+=['','## 分项时间','',
            '各分项独立取中位数 (s)。CUDA 后端的候选、质量、包围盒、与已接受单元重叠四项均含该次主机与显卡之间的搬移和同步；它们是调用墙钟时间。前沿合法性检查为 CPU 的域归属、三角面接触及必要 LP 检查。剩余部分含候选邻域排序、选择、调度与 Python 循环。','',
            '| 输入 | 后端 | 初始化 | 前沿几何合法性 | 拓扑更新 | 候选 | 质量 | 同批冲突 | 与已有单元重叠 |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,case in r['cases'].items():
        for b in ('cpu','cuda'):
            s=case[b].get('median_stages_s')
            if s:
                vals=[s.get(k,0.) for k in ('backend_initialization','front_geometric_legality','front_topology','backend_propose_local','backend_evaluate','backend_conflicts','backend_overlaps')]
                lines.append('| '+name+' | '+b+' | '+' | '.join(f'{x:.3f}' for x in vals)+' |')
    stress=out/'sphere-stress/cuda/run.json'
    if stress.exists():
        s=json.loads(stress.read_text()); lines+=['','## 较细球域单次压力试验','',
            '输入为 128 个三角面的球形多面体，局部目标长度 0.32 m，新点上限 64。']
        if s['status']=='ok':
            lines+=[f"成功生成并验证 {s['output_tets']} 个四面体（{s['output_vertices']} 点），AFT {s['aft_tets']} 个、CPU 残腔补全 {s['cpu_completion_tets']} 个，生成 {s['generation_s']:.3f} s，独立验证 {s['independent_validation_s']:.3f} s。"]
        elif s['status']=='timeout':
            last=s.get('last_work_snapshot',{})
            lines+=[f"达到 {s['timeout_s']:.0f} s 生成上限。最后一次进度采集在 {last.get('elapsed_generation_s',0.):.1f} s，已有 {last.get('provisional_aft_tets','未知')} 个暂接受四面体及 {last.get('residual_front_faces','未知')} 个残余前沿面。本次未取得完整网格。"]
        else: lines+=[f"生成状态 {s['status']}。详见 sphere-stress/cuda/run.json。"]
    lines+=['','CPU/CUDA 成对结果、重复次数、逐次完整时间及交叉后端连通性对照见 [summary.json](summary.json)。每个 run1 均附有独立几何检查。','',
            '重现命令（项目根目录）','','```powershell',
            '.\\.venv\\Scripts\\python.exe -m fem_comparison.benchmark_general --repeats 3 --timeout 65',
            '.\\.venv\\Scripts\\python.exe -m fem_comparison.benchmark_general --stress --timeout 120',
            '.\\.venv\\Scripts\\python.exe -m fem_comparison.benchmark_general --report','```']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--worker',nargs=2,metavar=('CASE','BACKEND'))
    parser.add_argument('--cases',nargs='+',default=['cube-small','cube-medium','hollow-large'],choices=list(CASES))
    parser.add_argument('--stress',action='store_true',help='Run one CUDA 128-face sphere within the timeout')
    parser.add_argument('--report',action='store_true',help='Summarize completed records without recomputing them')
    parser.add_argument('--prism-worker',choices=['cpu','cuda'])
    parser.add_argument('--prism-pair',action='store_true',help='One CPU/CUDA pair on the same supplied layered U seed')
    parser.add_argument('--mesh',type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--timeout',type=float,default=65.)
    args=parser.parse_args()
    args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    if (args.prism_worker or args.prism_pair) and args.mesh is None:parser.error('--mesh is required for prism mode')
    if args.prism_pair:prism_pair(args.mesh,args.out,args.timeout)
    elif args.prism_worker:prism_worker(args.mesh,args.prism_worker,args.out)
    elif args.report:write_report(args.out)
    elif args.worker:worker(*args.worker,args.out)
    elif args.stress:
        result=invoke_worker('sphere-stress','cuda',args.out/'sphere-stress'/'cuda',args.timeout,validate=True)
        print(json.dumps(result),flush=True)
    else:experiment(args.out,args.repeats,args.timeout,args.cases)
