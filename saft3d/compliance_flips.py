"""Fixed-node connectivity changes driven by the current P1 displacement.

For the same boundary traces and nodal displacement, a decrease in internal
energy is a computable lower bound on the improvement of the equilibrium
compliance. Boundary facets and all vertex coordinates are held fixed.
"""
from itertools import combinations
import numpy as np
from .topology_optimization import _adjacency, _candidate_patch
from fem_comparison.elasticity import solve, YOUNG, POISSON


def _energy(points, cells, displacement):
    xyz = points[cells]
    edges = xyz[:, 1:] - xyz[:, :1]
    volumes = np.abs(np.linalg.det(edges)) / 6
    good = volumes > 1.e-13
    energy = np.full(len(cells), np.inf)
    du = displacement[cells[:, 1:]] - displacement[cells[:, :1]]
    grad = np.linalg.solve(edges[good], du[good])
    eps = .5 * (grad + grad.transpose(0, 2, 1))
    shear = YOUNG / (2*(1+POISSON))
    lame = 2*shear*POISSON/(1-2*POISSON)
    energy[good] = volumes[good] * (2*shear*np.einsum('eij,eij->e',eps,eps)
                                   + lame*np.trace(eps, axis1=1,axis2=2)**2)
    sq = sum(np.sum((xyz[:, i]-xyz[:, j])**2,axis=1) for i,j in combinations(range(4),2))
    return energy, volumes, 12*np.cbrt(3*volumes)**2/sq


def _rings(cells):
    faces, edges = _adjacency(cells)
    for (i,j,k),ids in faces.items():
        if len(ids)!=2: continue
        a, = set(cells[ids[0]])-{i,j,k}
        b, = set(cells[ids[1]])-{i,j,k}
        if tuple(sorted((a,b))) not in edges:
            yield ids, ((a,b,i,j),(a,b,j,k),(a,b,k,i))
    for (a,b),ids in edges.items():
        if len(ids) not in (3,4): continue
        links=[set(cells[i])-{a,b} for i in ids]
        nodes=sorted(set.union(*links))
        if len(nodes)!=len(ids) or any(sum(v in link for link in links)!=2 for v in nodes):
            continue
        if len(nodes)==3:
            i,j,k=nodes
            if tuple(nodes) not in faces:
                yield ids, ((i,j,k,a),(i,j,k,b))
        else:
            # Each pair of opposite ring vertices is a possible new diagonal.
            for i,k in combinations(nodes,2):
                if {i,k} in links or (i,k) in edges: continue
                j,l=sorted(set(nodes)-{i,k})
                yield ids, ((i,k,a,j),(i,k,j,b),(i,k,b,l),(i,k,l,a))


def energy_flips(points,tets,boundary,passes=6):
    cells=tets.copy()
    metric,field=solve(points,cells,boundary)
    start=metric['compliance_J']
    history=[]
    for iteration in range(passes):
        u=field['displacement']
        old_energy,old_vol,old_q=_energy(points,cells,u)
        candidates=[]
        groups={}
        for ids,patch in _rings(cells):
            groups.setdefault((len(ids),len(patch)),[]).append((ids,patch))
        for (nold,nnew),group in groups.items():
            ids=np.asarray([ids for ids,_ in group])
            proposed=np.asarray([p for _,p in group])
            e,vol,q=_energy(points,proposed.reshape(-1,4),u)
            gain=old_energy[ids].sum(axis=1)-e.reshape(-1,nnew).sum(axis=1)
            v=old_vol[ids].sum(axis=1)
            valid=(gain>1.e-9*start) & (np.abs(v-vol.reshape(-1,nnew).sum(axis=1))<1.e-10*v)
            valid &= q.reshape(-1,nnew).min(axis=1)>np.maximum(.002, np.minimum(.10,.45*old_q[ids].min(axis=1)))
            for index in np.flatnonzero(valid):
                candidates.append((float(gain[index]),ids[index],proposed[index]))
        touched=set()
        additions=[]
        predicted=0.
        for gain,ids,patch in sorted(candidates,key=lambda c:-c[0]):
            if touched.intersection(ids): continue
            checked,reason=_candidate_patch(points,cells[ids],patch,min_gain=-np.inf)
            if checked is None: continue
            touched.update(ids); additions.append(checked);predicted+=gain
        if not additions: break
        new_cells=np.vstack([np.delete(cells,sorted(touched),axis=0),*additions])
        new_metric,new_field=solve(points,new_cells,boundary)
        actual=new_metric['compliance_J']-metric['compliance_J']
        if actual+start*1.e-8<predicted:
            raise ArithmeticError('Equilibrium gain fell below the fixed-displacement bound')
        if new_metric['free_dofs']!=metric['free_dofs']:raise ValueError('Changed free DOFs')
        cells,metric,field=new_cells,new_metric,new_field
        history.append({'pass':iteration+1,'patches':len(additions),
                        'predicted_gain_J':predicted,'actual_gain_J':actual,
                        'mean_loaded_ux_m':metric['mean_loaded_ux_m']})
    return cells,metric,field,history
