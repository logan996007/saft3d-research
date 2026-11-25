"""Paired CPU/CUDA scale experiment for the direct extrusion control.

Run in an otherwise idle local session. Each measured sample runs in its own
process. Device/kernel warm-up is reported separately. The timed region gets
the same coarse U-shaped boundary and returns the full mesh on the host.
"""
import argparse
import ctypes
from ctypes import wintypes
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import numpy as np

from saft3d.general_surfaces import make_general_surface
from saft3d.thickness_front import grow_thickness_front
from saft3d.validation import validate_mesh
from saft3d.io import write_vtk


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'output'/'scale-speed'/'directed'
MIB = 1024**2


class MemoryStatus(ctypes.Structure):
    _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)] + [
        (k,ctypes.c_ulonglong) for k in ('total_phys','avail_phys','total_pf','avail_pf',
                                          'total_virt','avail_virt','ext_virt')]


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb',wintypes.DWORD),('faults',wintypes.DWORD)] + [
        (k,ctypes.c_size_t) for k in ('peak_rss','rss','peak_paged','paged','peak_nonpaged',
                                      'nonpaged','pf','peak_pf')]


def available_ram():
    if os.name=='nt':
        status=MemoryStatus()
        status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('failed to read free host memory')
        return status.avail_phys
    raise OSError('The guarded benchmark currently requires Windows memory counters')


def process_mem(pid):
    if os.name=='nt':
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=wintypes.HANDLE
        handle=kernel.OpenProcess(0x410,False,pid)
        if not handle:return None
        try:
            pm=ProcessMemory();pm.cb=ctypes.sizeof(pm)
            if ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(handle),ctypes.byref(pm),pm.cb):
                return pm.peak_rss
        finally:kernel.CloseHandle(wintypes.HANDLE(handle))
        return None
    path=Path(f'/proc/{pid}/status')
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('VmHWM:'):return 1024*int(line.split()[1])
    return None


def available_vram():
    proc=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],
                        capture_output=True,check=True,text=True,timeout=15)
    return int(proc.stdout.splitlines()[0])*MIB


class GpuMemoryMonitor:
    """Read-only NVML query for device-wide use, including other applications."""
    def __init__(self):
        self.lib=ctypes.CDLL('nvml.dll')
        if self.lib.nvmlInit_v2():raise OSError('NVML initialization failed')
        self.device=ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0,ctypes.byref(self.device)):
            raise OSError('NVML device unavailable')

    def free(self):
        # nvmlMemory_t layout: total, free, used (all unsigned long long).
        storage=(ctypes.c_ulonglong*3)()
        if self.lib.nvmlDeviceGetMemoryInfo(self.device,ctypes.byref(storage)):
            raise OSError('NVML memory query failed')
        return int(storage[1])

    def close(self):
        self.lib.nvmlShutdown()


def approx_bytes(r,nlayers):
    cap=6*4**r
    cells=3*cap*nlayers;bound=2*cap+16*2**r*nlayers
    points=((cap+8*2**r)//2+1)*(nlayers+1)
    # Conservative working-set envelope including two output-sized scratch
    # buffers, Python/runtime overhead, GPU drivers and independent checker.
    return (900*MIB+cells*88+points*112+bound*160,
            480*MIB+cells*36+points*48)


def verify_sweep(points,cells,boundary,r,layers):
    """Independent staircase incidence, all-column coordinates, wall and volume checks."""
    ncap=6*4**r; perimeter=8*2**r
    if len(cells)!=3*ncap*layers or len(boundary)!=2*ncap+2*perimeter*layers:
        raise RuntimeError('output count mismatch')
    bottom=np.flatnonzero(points[:,2]==-.5)
    top=np.flatnonzero(points[:,2]==.5)
    n=len(bottom)
    if len(points)!=(layers+1)*n or n!=(ncap+perimeter)//2+1:
        raise RuntimeError('Euler count or number of layer vertices differs')
    if not np.array_equal(points[top,:2],points[bottom,:2]):raise RuntimeError('unmatched caps')
    bottom_faces=boundary[:ncap,::-1]
    cap=np.searchsorted(bottom,bottom_faces)
    if (not np.array_equal(bottom[cap],bottom_faces)
            or not np.array_equal(top[cap],boundary[ncap:2*ncap])):
        raise RuntimeError('cap triangle maps differ')
    xy=points[bottom,:2]
    tri=xy[cap]
    u,v=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]
    area=(u[:,0]*v[:,1]-u[:,1]*v[:,0])/2
    if area.min()<=0 or abs(area.sum()-7.)>1.e-10:raise RuntimeError('invalid U cross-section')
    # Every refined triangle must lie in one original cap triangle. Their
    # signed areas partition each original triangle. Check the cap edge chain.
    v0,f0,_=make_general_surface('u_prism')
    original=v0[f0[(v0[f0,2]==.5).all(axis=1)],:2]
    membership=np.zeros(ncap,dtype=np.int32)
    for corners in original:
        side=np.roll(corners,-1,axis=0)-corners
        inside=np.ones(ncap,bool)
        for anchor,vector in zip(corners,side):
            d=tri-anchor
            inside &= (vector[0]*d[:,:,1]-vector[1]*d[:,:,0]).min(axis=1)>=-1.e-13
        membership+=inside
        if abs(area[inside].sum()-np.abs(np.linalg.det(corners[1:]-corners[:1]))/2)>1.e-10:
            raise RuntimeError('cap area does not match original facet')
    if np.any(membership!=1):raise RuntimeError('cap coverage mismatch')
    edge=cap[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
    key=edge.min(axis=1)*n+edge.max(axis=1)
    _,idx,inv,num=np.unique(key,return_index=True,return_inverse=True,return_counts=True)
    net=np.bincount(inv,weights=np.where(edge[:,0]<edge[:,1],1,-1))
    if len(idx[num==1])!=perimeter or np.max(num)!=2 or np.any(net[num==2]):
        raise RuntimeError('cap is not an oriented manifold')
    perimeter_edges=edge[idx[num==1]]
    a,b,c=np.sort(cap,axis=1).T
    def ids(i):
        if i==0:return bottom
        if i==layers:return top
        return np.arange(2*n+(i-1)*n,2*n+i*n)
    blocks=cells.reshape(layers,ncap,3,4)
    wall=boundary[2*ncap:].reshape(layers,2*perimeter,3)
    def tri_codes(faces):
        s=np.sort(faces,axis=1)
        # Lexicographic comparison avoids packing three big vertex indices
        # into an overflowing fixed-width integer key.
        return np.lexsort((s[:,2],s[:,1],s[:,0])),s
    for i,block in enumerate(blocks):
        lo,hi=ids(i),ids(i+1)
        if not np.array_equal(points[hi,:2],xy) or np.max(np.abs(points[hi,2]-(float(i+1)/layers-.5)))>1.e-13:
            raise RuntimeError('wrong layer coordinates')
        # Sorted corner sets verify all three positive stair simplices in every
        # prism independently of the orientation swap used by the generator.
        for j,tr in enumerate(((lo[a],lo[b],lo[c],hi[c]),(lo[a],lo[b],hi[b],hi[c]),
                               (lo[a],hi[a],hi[b],hi[c]))):
            if not np.array_equal(np.sort(np.stack(tr,axis=1),axis=1),np.sort(block[:,j],axis=1)):
                raise RuntimeError('incorrect staircase tetrahedron')
        # Verify exposed 2D perimeter is swept on the same oriented side patch.
        x,y=perimeter_edges.T
        side1=np.column_stack((lo[x],lo[y],np.where(x<y,hi[y],hi[x])))
        side2=np.column_stack((np.where(x<y,lo[x],lo[y]),hi[y],hi[x]))
        expected=np.vstack((side1,side2));observed=wall[i]
        ix,sx=tri_codes(expected);iy,sy=tri_codes(observed)
        if not np.array_equal(sx[ix],sy[iy]):raise RuntimeError('wrong sidewall triangles')
        # Preserve orientation as well as the unordered triangle keys.
        def parity(triangles):
            a,b,c=triangles.T
            return (a>b)^(a>c)^(b>c)
        if not np.array_equal(parity(expected)[ix],parity(observed)[iy]):
            raise RuntimeError('wrong sidewall orientation')
    return {'all_prisms_checked':int(ncap*layers),'all_cells_checked':len(cells),
            'original_cap_facets_matched':True,'manifold_cap':True,'all_wall_triangles_matched':True,
            'all_layer_coordinates_matched':True,
            'argument':'verified cap partition swept through increasing disjoint z intervals, same staircase on each shared prism face'}


def trial(engine,r,layers,export=None):
    v,f,_=make_general_surface('u_prism')
    fractions=np.linspace(0.,1.,layers+1)
    start=time.perf_counter()
    p,t,b,rep=grow_thickness_front(v,f,fractions,backend=engine,mode='sweep',cap_refinements=r)
    elapsed=time.perf_counter()-start
    check_start=time.perf_counter()
    certificate=verify_sweep(p,t,b,r,layers)
    digest={name:hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()
            for name,a in [('points',p),('cells',t),('boundary',b)]}
    result={'backend':engine,'cap_refinements':r,'layers':layers,'tetrahedra':len(t),
            'cap_triangles':6*4**r,'points':len(p),'boundary_faces':len(b),'elapsed_s':elapsed,
            'stages':rep['stage_seconds'],'device':rep['device'],'raw_result_hashes':digest,
            'volume':rep['total_volume'],'min_cell_volume':rep['min_volume'],
            'min_quality':rep['min_mean_ratio'],'certificate':certificate,'verification_and_hash_s':time.perf_counter()-check_start,
            'mesh_bytes':p.nbytes+t.nbytes+b.nbytes}
    if engine=='cuda':
        import cupy as cp
        result['cupy_pool_reserved_bytes']=cp.get_default_memory_pool().total_bytes()
    if r==0:
        validation=validate_mesh(p,t,b)
        if not validation['valid']:raise RuntimeError(validation['errors'])
        result['independent_full_LP_passed']=True
    if export is not None:
        write_vtk(export,p,t,title='U-shaped solid: validated direct extrusion / tetrahedra')
        result['export_vtk']=export.name
    actual_peak=process_mem(os.getpid())
    if actual_peak is not None and actual_peak<result['mesh_bytes']:
        raise RuntimeError('measured peak RSS is smaller than resident mesh; reject memory measurement')
    result['worker_self_peak_rss_bytes']=actual_peak
    return result


def worker(engine,r,layers,reps,output):
    output.with_suffix('.pid').write_text(str(os.getpid()),encoding='ascii')
    cold=trial(engine,r,layers,output.with_name('u_sweep_73728.vtk') if r==4 and engine=='cuda' else None)
    workspace={'backend':engine,'worker_pid':os.getpid(),'cold_process_first_call':cold,'samples':[]}
    for _ in range(reps):
        gc.collect()
        result=trial(engine,r,layers)
        workspace['samples'].append(result)
        workspace['median_elapsed_s']=statistics.median(x['elapsed_s'] for x in workspace['samples'])
        output.write_text(json.dumps(workspace,indent=2),encoding='utf-8')


def isolated(engine,r,layers,reps,output,timeout):
    free=available_ram();free_gpu=available_vram() if engine=='cuda' else None
    ram,device=approx_bytes(r,layers);rss_budget=.5*free
    if ram>rss_budget or (free_gpu is not None and device>.7*free_gpu):
        return {'stopped':'estimated working set exceeds resource budget','ram_estimate':ram,'available_ram':free,
                'device_estimate':device,'available_vram':free_gpu}
    command=[sys.executable,'-m','fem_comparison.benchmark_scale','--worker',engine,'--r',str(r),
             '--layers',str(layers),'--reps',str(reps),'--output',str(output)]
    flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    monitor=None
    if engine=='cuda':
        try:monitor=GpuMemoryMonitor()
        except (AttributeError,OSError) as exc:return {'stopped':f'live NVML guard unavailable: {exc}'}
    min_vram=free_gpu
    # Windows venv executables can be redirectors. The actual interpreter
    # writes its PID before any mesh allocations, so the guardian watches it.
    pidfile=output.with_suffix('.pid')
    pidfile.unlink(missing_ok=True)
    with subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
                          creationflags=flags,env=dict(os.environ,SAFT_SCALE_GUARDIAN_PID=str(os.getpid()))) as process:
        t=time.perf_counter();peak=0;min_free=free;reason=None;actual_pid=None
        while process.poll() is None:
            if actual_pid is None and pidfile.is_file():
                text=pidfile.read_text(encoding='ascii').strip()
                if text:actual_pid=int(text)
            now=available_ram();rss=process_mem(actual_pid or process.pid) or 0
            peak=max(peak,rss);min_free=min(min_free,now)
            if time.perf_counter()-t>timeout:reason='wall-clock timeout'
            if now<max(2*1024*MIB,.4*free) or rss>rss_budget:reason='host memory guard'
            if monitor:
                try:min_vram=min(min_vram, monitor.free())
                except OSError:
                    monitor.close();monitor=None;reason='live NVML guard became unavailable'
                if min_vram<max(512*MIB,.18*free_gpu):reason='device memory guard'
            if reason:
                if actual_pid is not None and actual_pid!=process.pid:
                    try:
                        import signal
                        os.kill(actual_pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                process.kill();break
            time.sleep(.2)
        stdout,stderr=process.communicate()
        pidfile.unlink(missing_ok=True)
        if monitor:monitor.close()
        if reason or process.returncode!=0:
            return {'stopped':reason or 'worker failed','exit':process.returncode,'stderr':stderr[-3000:],
                    'peak_working_set_bytes':peak,'min_available_ram_bytes':min_free}
    data=json.loads(output.read_text(encoding='utf8'))
    peak=max([peak]+[x['worker_self_peak_rss_bytes'] or 0 for x in [data['cold_process_first_call'],*data['samples']]])
    data.update({'worker_peak_rss_bytes':peak,'available_ram_initial_bytes':free,'min_free_host_bytes':min_free,
                 'available_vram_initial_bytes':free_gpu,'min_free_device_bytes_sampled':min_vram,
                 'nvml_live_monitor':monitor is not None})
    output.write_text(json.dumps(data,indent=2),encoding='utf8')
    return data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',choices=['cpu','cuda']);parser.add_argument('--r',type=int,default=0)
    parser.add_argument('--layers',type=int,default=2);parser.add_argument('--reps',type=int,default=3)
    parser.add_argument('--output',type=Path,default=OUT/'trial.json');parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--levels',type=int,nargs='+',default=[2,3,4,5],
                        help='cap refinement counts (default: through 589824 cells); 6/7 opt into large runs')
    parser.add_argument('--timeout',type=int,default=240)
    parser.add_argument('--large-reps',type=int,default=1)
    parser.add_argument('--layer-factor',type=float,default=1.)
    args=parser.parse_args()
    if os.name!='nt':
        print('SKIPPED: guarded timing requires Windows process/host memory counters and NVIDIA NVML. '
              'The thickness-front generator itself is independently usable on other supported platforms.')
        return
    if (min(args.levels)<0 or max(args.levels)>9 or args.reps<1 or args.large_reps<1
            or not np.isfinite(args.layer_factor) or not 0<args.layer_factor<=8 or args.timeout<=0):
        parser.error('invalid levels, repetitions, layer factor, or timeout')
    if args.worker:
        if not os.environ.get('SAFT_SCALE_GUARDIAN_PID'):
            parser.error('--worker is internal; use --levels so OS memory/time guards are active')
        worker(args.worker,args.r,args.layers,args.reps,args.output)
        return
    args.out.mkdir(parents=True,exist_ok=True)
    suite={'mode':'U-shaped z-extrusion direct staircase sweep, CPU NumPy vs CUDA, float64 geometry/int64 cells',
           'input_surface':'16 vertices, 28 triangles; cap refinement and layers scaled together',
           'timing_scope':'warm public generator, validated coarse PLC, host output, positive-cell and quality checks included; separate certificate + SHA excluded',
           'host_cpu':os.getenv('PROCESSOR_IDENTIFIER'),'python':sys.version.split()[0],'numpy':np.__version__,
           'cases':[]}
    for r in args.levels:
        layers=max(1,round(2**r*args.layer_factor))
        row={'refinements':r,'layers':layers,'expected_tets':18*4**r*layers}
        # Decline the whole pair before allocating either large CPU or GPU
        # output if the target configuration already exceeds a safe budget.
        try:
            free_ram,free_gpu=available_ram(),available_vram()
            monitor=GpuMemoryMonitor();monitor.free();monitor.close()
        except (OSError,AttributeError,subprocess.SubprocessError,ValueError) as exc:
            print(f'SKIPPED: paired benchmark needs working host/GPU memory monitors: {exc}')
            return
        ram,dev=approx_bytes(r,layers)
        if ram>.5*free_ram or dev>.7*free_gpu:
            row['stopped']='pair preflight exceeds host or GPU budget'
            row['capacity_bytes']=dict(host_estimate=ram,device_estimate=dev,
                                       available_host=free_ram,available_device=free_gpu)
            suite['cases'].append(row)
            (args.out/'summary.json').write_text(json.dumps(suite,indent=2),encoding='utf8')
            print(f"SKIPPED {row['expected_tets']}: pair exceeds memory budget",flush=True)
            continue
        for engine in ('cpu','cuda'):
            file=args.out/f'r{r}-{engine}.json'
            data=isolated(engine,r,layers,args.reps if r<=6 else args.large_reps,file,args.timeout)
            row[engine]=({k:v for k,v in data.items() if k not in ('samples',)} if 'stopped' in data
                         else {'raw':file.name,'median_s':data['median_elapsed_s'],'samples':len(data['samples']),
                               'cold_first_call_s':data['cold_process_first_call']['elapsed_s'],
                               'peak_rss_bytes':data['worker_peak_rss_bytes'],'min_quality':data['samples'][0]['min_quality'],
                               'hashes':data['samples'][0]['raw_result_hashes']})
            if 'stopped' in data:break
        if 'cuda' in row and 'stopped' not in row['cuda']:
            row['bitwise_identical']=row['cpu']['hashes']==row['cuda']['hashes']
            if not row['bitwise_identical']:raise RuntimeError('CPU and CUDA produced different meshes')
            row['speedup_cpu_over_gpu']=row['cpu']['median_s']/row['cuda']['median_s']
        suite['cases'].append(row)
        (args.out/'summary.json').write_text(json.dumps(suite,indent=2),encoding='utf8')
        print(r,row['expected_tets'],row.get('speedup_cpu_over_gpu','stopped'),flush=True)


if __name__=='__main__':main()
