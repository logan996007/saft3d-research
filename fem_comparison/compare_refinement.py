"""Read the generated tetrahedra, verify matching budgets, solve P1, and plot errors."""
import hashlib
import json
from pathlib import Path
import numpy as np
from . import elasticity, thermal
from .run import ROOT, vtk_fields
from saft3d.io import write_json


OUT = ROOT / 'output' / 'local-refinement'
LEVELS = {'u': (156, 966, 3078), 's': (48, 96, 192)}


def read_npz(path):
    with np.load(path, allow_pickle=False) as mesh:
        return tuple(mesh[k] for k in ('vertices', 'tetrahedra', 'boundary_faces'))


def verify_same_nodes(mesh, control):
    points, tets, boundary = mesh
    other, other_tets, other_boundary = control
    used = np.unique(tets)
    if not np.array_equal(points[used], other[np.unique(other_tets)]):
        raise ValueError('Same-node control changed the set of active coordinates')
    def keys(tris):
        return {tuple(row) for row in np.sort(tris, axis=1)}
    if keys(np.searchsorted(used, boundary)) != keys(np.searchsorted(np.unique(other_tets), other_boundary)):
        raise ValueError('Same-node control changed boundary triangles')


def compare():
    baseline = json.loads((ROOT/'output/fem-comparison/results.json').read_text(encoding='utf8'))
    reference = baseline['independent_elasticity']['u_finer_p2']['mean_loaded_ux_m']
    fields_dir = OUT/'fields'
    fields_dir.mkdir(exist_ok=True, parents=True)
    rows = []
    for problem, levels in LEVELS.items():
        for i, budget in enumerate(levels):
            paths = {'saft':OUT/problem/f'level{i}.npz','cpu':OUT/'cpu'/f'{problem}-{budget}.npz',
                     'cpu_same':OUT/'cpu'/f'{problem}-{budget}-same.npz'}
            meshes = {variant:read_npz(path) for variant,path in paths.items()}
            verify_same_nodes(meshes['saft'], meshes['cpu_same'])
            agent_report=json.loads(paths['saft'].with_suffix('.json').read_text(encoding='utf8'))
            if problem=='s':
                assert agent_report['validation']['valid'] and agent_report['generator']['backend']=='cuda'
                assert agent_report['generator']['mode']=='aft'
            else:
                assert agent_report['mesh_validation']['valid']
            for variant,(p,t,f) in meshes.items():
                case = f'{problem}-{budget}-{variant}'
                q,fields = (elasticity if problem=='u' else thermal).solve(p,t,f)
                if q['free_dofs']!=budget:
                    raise RuntimeError(f'{case}: expected {budget} free DOFs, got {q["free_dofs"]}')
                expected_volume = 7.0 if problem=='u' else .9936
                coords = p[t]
                volumes = np.linalg.det(coords[:,1:]-coords[:,:1])/6
                if volumes.min()<=0 or not np.isclose(volumes.sum(),expected_volume,atol=1e-10,rtol=1e-10):
                    raise ValueError(f'{case}: invalid volume')
                record = dict(problem=problem,variant=variant,case=case,reference_free_budget=budget,
                              input=paths[variant].relative_to(ROOT).as_posix(),
                              sha256=hashlib.sha256(paths[variant].read_bytes()).hexdigest(),**q)
                if problem=='u':
                    record['relative_ux_error'] = abs(q['mean_loaded_ux_m']/reference-1.)
                    if q['force_balance_relative']>1e-8: raise ValueError(f'{case}: force imbalance')
                    nodes={'displacement_m':fields['displacement']}; cells={'von_mises_Pa':fields['von_mises']}
                else:
                    if q['relative_heat_imbalance']>1e-9: raise ValueError(f'{case}: heat imbalance')
                    nodes={'temperature_K':fields['temperature']}; cells={'heat_flux_W_m2':fields['heat_flux']}
                vtk_fields(fields_dir/f'{case}.vtk',p,t,nodes,cells)
                rows.append(record)
    result={'reference_displacement_m':reference, 'meshes':rows,
            'comparison':'P1; DOFs counted after Dirichlet elimination; CPU same-node cases also preserve boundary facets'}
    write_json(OUT/'comparison.json',result)
    return rows,baseline,reference


def plot_errors(rows, baseline, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if 'Microsoft YaHei' in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8),layout='constrained')
    palette={'cpu':'#bf6918','cpu_same':'#616488','saft':'#0b716e'}
    for ax,problem in zip(axes,('u','s')):
        budgets=LEVELS[problem]
        key='relative_ux_error' if problem=='u' else 'relative_l2_temperature_rise'
        for variant,label,style in [('cpu','CPU TetGen 常规','s--'),('cpu_same','CPU TetGen 同节点','D:'),('saft','局部加密 SAFT','o-')]:
            group=[r for r in rows if r['problem']==problem and r['variant']==variant]
            ax.plot(budgets,[r[key]*100 for r in group],style,color=palette[variant],lw=1.8,label=label,
                    mfc='none' if variant!='saft' else palette[variant],ms=9 if variant=='cpu_same' else 5)
        old=baseline['meshes'][f'{problem}_saft']
        previous=abs(old['mean_loaded_ux_m']/reference-1.) if problem=='u' else old[key]
        ax.scatter([budgets[0]],[100*previous],s=54,c='#303030',marker='x',label='原 SAFT 粗网格')
        ax.set_xscale('log');ax.set_xticks(budgets, [str(d) for d in budgets]);ax.minorticks_off()
        if problem=='s':ax.set_yscale('log')
        ax.set_xlabel('未约束的一阶自由度');ax.set_ylabel('相对误差 (%)')
        ax.set_title('U 形件加载端平均位移' if problem=='u' else 'S 形件温升场 · 体积 L2 范数')
        ax.grid(True,alpha=.22);ax.margins(x=.12,y=.17)
    handles, labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.savefig(OUT/'refinement-errors.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    plot_errors(*compare())
