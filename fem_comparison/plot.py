"""Plot physical sections from the actual ParaView field exports, on shared scales."""
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import font_manager
from .run import OUT


def read_field(case):
    tokens=(OUT/'fields'/f'{case}.vtk').read_text('ascii').split()
    def block(marker, skip, width, dtype):
        offset=tokens.index(marker)
        count=int(tokens[offset+1]); start=offset+skip
        return np.array(tokens[start:start+count*width],dtype=dtype).reshape(count,width)
    p=block('POINTS',3,3,float); cells=block('CELLS',3,5,int)[:,1:]
    field=tokens.index('POINT_DATA')+2
    if tokens[field]=='VECTORS':
        values=np.array(tokens[field+3:field+3+3*len(p)],float).reshape(-1,3)
    else:
        start=tokens.index('LOOKUP_TABLE',field)+2
        values=np.array(tokens[start:start+len(p)],float)
    return p,cells,values


def thermal_section(p,cells,temp,y=.15):
    coords,values,triangles=[],[],[]
    for ids in cells:
        tet=p[ids]; d=tet[:,1]-y
        if d.min()>0 or d.max()<0: continue
        section=[(tet[i,[0,2]],temp[ids[i]]) for i in range(4) if d[i]==0]
        for a,b in combinations(range(4),2):
            if d[a]*d[b]>=0: continue
            r=d[a]/(d[a]-d[b])
            section.append(((tet[a]+r*(tet[b]-tet[a]))[[0,2]],temp[ids[a]]+r*(temp[ids[b]]-temp[ids[a]])))
        if len(section)<3:continue
        pts=np.array([pair[0] for pair in section]); rel=pts-pts.mean(axis=0)
        order=np.argsort(np.arctan2(rel[:,1],rel[:,0])); start=len(coords)
        coords.extend(pts[order]);values.extend(np.array([pair[1] for pair in section])[order]-300.)
        for i in range(1,len(pts)-1):triangles.append([start,start+i,start+i+1])
    return np.array(coords),np.array(triangles),np.array(values)


def plot_fields():
    if 'Microsoft YaHei' in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    report=json.loads((OUT/'results.json').read_text(encoding='utf8'))['meshes']
    fig,axes=plt.subplots(2,3,figsize=(12.6,6.2),layout='constrained',height_ratios=[1.15,.85])
    fig.suptitle('同一物理问题在不同四面体网格上的一阶有限元解',fontsize=17,fontweight='bold')
    for row, cases in enumerate([['u_saft','u_coarse','u_finer'],['s_saft','s_coarse','s_finer']]):
        subset=[]
        for case in cases:
            p,t,field=read_field(case)
            if row==0:
                on_top=np.isclose(p[t,2],.5)
                mask=on_top.sum(axis=1)==3
                faces=t[mask][on_top[mask]].reshape(-1,3)
                subset.append((p[:,:2],faces,np.linalg.norm(field,axis=1)*1e3))
            else: subset.append(thermal_section(p,t,field))
        vmax=max(vals[np.unique(tris)].max() for _,tris,vals in subset)
        norm=Normalize(0,max(vmax,7.2 if row==1 else 0.));cmap='viridis' if row==0 else 'inferno'
        for ax,(coords,tris,vals),case,title in zip(axes[row],subset,cases,['SAFT','CPU TetGen · 同自由度','CPU TetGen · 加密']):
            ax.tripcolor(coords[:,0],coords[:,1],tris,vals,shading='gouraud',cmap=cmap,norm=norm)
            if row==0:ax.set_aspect('equal');ax.set_xlim(-.08,3.08);ax.set_ylim(-.08,3.08)
            else: ax.set_xlim(0,4);ax.set_ylim(-.12,.12);ax.set_yticks([-.12,0,.12]);ax.set_box_aspect(.42)
            ax.set_title(f"{title}  /  {report[case]['free_dofs']:,} 自由度",fontsize=10)
            ax.set_xlabel('x (m)');ax.set_ylabel('y (m)' if row==0 else 'z (m)')
            ax.tick_params(labelsize=8)
        label='位移模长 (mm)' if row==0 else '相对 300 K 的温升 (K)'
        fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),ax=axes[row,:],shrink=.84,label=label)
    fig.savefig(OUT/'comparison.png',dpi=155,facecolor='white');plt.close(fig)

if __name__=='__main__': plot_fields()
