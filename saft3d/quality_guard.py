"""Improve tetrahedral shape while guarding the U-bracket compliance and DOFs.

The fast treatment moves vertices tangentially to their geometric constraints.
The full treatment can also change connectivity and reallocate vertex IDs.
Both keep the original free-DOF budget. The reference displacement is used
only for post-optimization reporting.
"""
from __future__ import annotations
from itertools import combinations
from time import perf_counter
import numpy as np
from scipy.optimize import minimize
from .compliance_relocation import accumulate,free_coordinates,check_boundary_motion
from fem_comparison import elasticity

EDGES=np.array(list(combinations(range(4),2)))
FACES=np.array([(1,2,3),(0,3,2),(0,1,3),(0,2,1)])


def shape_data(points,cells):
    xyz=points[cells]; edges=xyz[:,EDGES[:,1]]-xyz[:,EDGES[:,0]]
    rawvol=np.einsum('ij,ij->i',edges[:,0],np.cross(edges[:,1],edges[:,2]))
    edge2=np.einsum('tej,tej->te',edges,edges); edge_sum=edge2.sum(axis=1)
    quality=12*np.cbrt(np.maximum(rawvol/2,0)**2)/edge_sum
    face_xyz=xyz[:,FACES]; outer=np.cross(face_xyz[:,:,1]-face_xyz[:,:,0],face_xyz[:,:,2]-face_xyz[:,:,0]); areas=np.linalg.norm(outer,axis=2)
    normals=outer/np.maximum(areas[:,:,None],1.e-30)
    angles=np.arccos(np.clip(-np.einsum('tej,tej->te',normals[:,EDGES[:,0]],normals[:,EDGES[:,1]]),-1,1))*180/np.pi
    return xyz,rawvol,edges,edge_sum,quality,areas,angles


def measures(points,cells):
    _,det,_,_,quality,_,angles=shape_data(points,cells)
    return {'mean_ratio':{'min':float(quality.min()),'p05':float(np.quantile(quality,.05)),'median':float(np.median(quality))},
            'dihedral_degrees':{'min':float(angles.min()),'p05_cell_min':float(np.quantile(angles.min(axis=1),.05)),
                                'median_cell_min':float(np.median(angles.min(axis=1))),'max':float(angles.max())},
            'min_signed_volume':float(det.min()/6)}


def quality_cost(points,cells,initial_det,goal):
    """Squared positive part of log(goal / mean ratio), with inversion extension."""
    xyz=points[cells]; e=xyz[:,1:]-xyz[:,:1];a,b,c=e[:,0],e[:,1],e[:,2]
    ca,cb,cc=np.cross(b,c),np.cross(c,a),np.cross(a,b)
    det=np.einsum('ij,ij->i',a,ca)
    floor=.02*initial_det;safe=np.maximum(det,floor)
    # Extend log(det) by its tangent below a small positive per-cell floor.
    logdet=np.log(safe)+np.minimum(det-floor,0)/floor
    ddet=np.stack((-ca-cb-cc,ca,cb,cc),axis=1)
    centred=xyz-xyz.mean(axis=1)[:,None,:]
    l2=np.maximum(4*np.einsum('tej,tej->t',centred,centred),1.e-30)
    log_quality=np.log(12.)+2./3*(logdet-np.log(2.))-np.log(l2)
    deficit=np.maximum(np.log(goal)-log_quality,0)
    dlogq=(2./3)*ddet/safe[:,None,None]-8*centred/l2[:,None,None]
    local=-deficit[:,None,None]*dlogq
    return .5*float(deficit@deficit),accumulate(local,cells,len(points))


def quality_guard(vertices,cells,boundary,*,loss_budget=.002,targets=(.06,.10,.16,.24,.32),iterations=90):
    """Accept quality-improving positions within a relative input-compliance budget.

    Accepted candidates must improve the minimum mean ratio AND minimum dihedral
    angle over the original mesh, prevent inversion, and preserve the physical
    boundary and active DOF count. A fixed target schedule sets geometric goals.
    Selection is lexicographic on minimum mean ratio and minimum dihedral angle.
    """
    if not np.isfinite(loss_budget) or not 0<=loss_budget<1:raise ValueError('invalid compliance budget')
    start=perf_counter();base=np.array(vertices,dtype=float);cells=np.asarray(cells);boundary=np.asarray(boundary)
    movable,_=free_coordinates(base,boundary)
    initial,fields=elasticity.solve(base,cells,boundary);C0=initial['compliance_J'];Cmin=C0*(1-loss_budget)
    before=measures(base,cells);det0=shape_data(base,cells)[1];assert (det0>0).all()
    best=base.copy(); best_shape=before;best_fem=initial;history=[]
    # Coordinates are expressed in units of the local mean edge scale.
    _,_,_,l2,*_=shape_data(base,cells)
    scale=np.bincount(cells.ravel(),weights=np.repeat(np.sqrt(l2/6),4),minlength=len(base))/np.maximum(np.bincount(cells.ravel(),minlength=len(base)),1)
    for goal in targets:
        if not 0<goal<=1:raise ValueError('mean ratio targets must lie in (0,1]')
        seed=best.copy(); old=shape_data(seed,cells)[4]
        active=np.unique(cells[old<1.30*goal]);mask=movable.copy();mask[~np.isin(np.arange(len(base)),active)]=False
        if not np.any(mask):continue
        def positions(x):
            change=np.zeros(base.shape);change[mask]=x
            return seed+change*scale[:,None]
        # Tikhonov displacement cost picks a nearby solution when many nodes
        # could fix the same bad element; it remains far smaller than deficits.
        def objective(x):
            pos=positions(x);value,g=quality_cost(pos,cells,det0,goal)
            return value+.01*(x@x),(g*scale[:,None])[mask]+.02*x
        result=minimize(objective,np.zeros(mask.sum()),method='L-BFGS-B',jac=True,
                        bounds=[(-.6,.6)]*int(mask.sum()),options={'maxiter':iterations,'ftol':1e-12,'gtol':1e-7,'maxls':40})
        opt=positions(result.x); selection=None
        for factor in (1.,.88,.75,.625,.50,.375,.25,.125,.0625,.03125):
            candidate=seed+factor*(opt-seed);shape=measures(candidate,cells)
            if shape['min_signed_volume']<=0 or shape['mean_ratio']['min']<=best_shape['mean_ratio']['min'] or shape['dihedral_degrees']['min']<=before['dihedral_degrees']['min']:continue
            try:check_boundary_motion(base,candidate,boundary)
            except RuntimeError:continue
            metrics,_=elasticity.solve(candidate,cells,boundary)
            if metrics['free_dofs']!=initial['free_dofs'] or metrics['compliance_J']<Cmin:continue
            selection=(candidate,metrics,shape);break
        entry={'target':goal,'active_coordinates':int(mask.sum()),'iterations':int(result.nit),'cost':float(result.fun),'accepted':selection is not None}
        if selection:
            best,best_fem,best_shape=selection;entry.update(fraction=factor,min_quality=best_shape['mean_ratio']['min'],min_angle=best_shape['dihedral_degrees']['min'],compliance=best_fem['compliance_J'])
        history.append(entry)
    report={'before':before,'after':best_shape,'elasticity_before':initial,'elasticity_after':best_fem,'history':history,
            'loss_budget':loss_budget,'measured_compliance_loss':(C0-best_fem['compliance_J'])/C0,'seconds':perf_counter()-start,
            'boundary_check':check_boundary_motion(base,best,boundary),'final_dofs':best_fem['free_dofs']}
    return best,report


def guarded_flips(points,tets,faces,compliance_floor,goal=.25,max_passes=12):
    """Quality-improving disjoint patch replacements with a compliance floor.

    For fixed vertices and patch boundary the nodal load is unchanged. Hence
    current C minus any increase in internal energy at the old displacement
    is a lower bound on the re-solved compliance after a batch of flips.
    """
    from .compliance_flips import _rings,_energy
    from .topology_optimization import _candidate_patch
    cells=tets.copy(); current,field=elasticity.solve(points,cells,faces);logs=[]
    for iteration in range(max_passes):
        u=field['displacement']; old_e,old_v,old_q=_energy(points,cells,u)
        groups={}; candidates=[]
        for ids,patch in _rings(cells):
            if min(old_q[ids])>=goal: continue
            groups.setdefault((len(ids),len(patch)),[]).append((ids,patch))
        for (_,k),group in groups.items():
            ids=np.asarray([i for i,_ in group]);patch=np.asarray([c for _,c in group]);new_e,new_v,new_q=_energy(points,patch.reshape(-1,4),u)
            gain=old_e[ids].sum(axis=1)-new_e.reshape(-1,k).sum(axis=1)
            q0=old_q[ids].min(axis=1); q1=new_q.reshape(-1,k).min(axis=1)
            valid=(q1>q0+.0001)&(abs(new_v.reshape(-1,k).sum(axis=1)-old_v[ids].sum(axis=1))<1.e-10*old_v[ids].sum(axis=1))
            for j in np.flatnonzero(valid):
                candidates.append((float(q0[j]),float(gain[j]),float(q1[j]-q0[j]),ids[j],patch[j]))
        touched=set();touched_vertices=set();added=[];predicted=0.;C=current['compliance_J']
        for _,gain,_,ids,patch in sorted(candidates,key=lambda c:(c[0],-c[1],-c[2])):
            if C+predicted+gain < compliance_floor:continue
            if touched_vertices.intersection(patch.ravel()):continue
            checked,_=_candidate_patch(points,cells[ids],patch,min_gain=0.)
            if checked is None:continue
            touched.update(ids);touched_vertices.update(patch.ravel());added.append(checked);predicted+=gain
        if not added:break
        proposal=np.vstack([np.delete(cells,sorted(touched),axis=0),*added])
        met,sol=elasticity.solve(points,proposal,faces)
        if met['compliance_J']<max(compliance_floor,C+predicted)-1.e-7 or met['free_dofs']!=current['free_dofs']:
            raise RuntimeError('fixed-displacement bound or free DOF budget failed')
        cells,current,field=proposal,met,sol
        logs.append({'iteration':iteration,'flips':len(added),'predicted_gain_J':predicted,'compliance_J':met['compliance_J'],'min_quality':float(shape_data(points,cells)[4].min())})
    return cells,logs


def joint_quality_compliance(vertices,cells,faces,reference_metrics,*,goal=.13,loss_budget=.002,iterations=160):
    """Relocate with a shape objective and an equilibrium-compliance penalty.

    The most uniform verified iterate strictly within the original compliance
    budget is retained. Temporary over-budget trial geometries are never exported.
    """
    from .compliance_relocation import gradient, local_geometry
    from scipy.optimize import minimize
    base=vertices.copy();refC=reference_metrics['compliance_J'];floor=refC*(1-loss_budget)
    geom=local_geometry(base,cells);initial_det=geom[2]*6
    m0,s0=elasticity.solve(base,cells,faces);movable,loaded=free_coordinates(base,faces)
    _,en,_=gradient(base,cells,loaded,s0['displacement'],geom)
    stiff=en/(geom[3]/6);nodes=np.bincount(cells.ravel(),weights=np.repeat(stiff,4),minlength=len(base)); typical=np.median(nodes[nodes>0]);scale=np.sqrt(typical/np.maximum(nodes,.04*typical)).clip(.1,4)
    seed_shape=measures(base,cells);best=base.copy();best_q=0.;hist=[];evaluation=0
    def objective(x):
        nonlocal best,best_q,evaluation
        evaluation+=1
        dx=np.zeros_like(base);dx[movable]=x; pos=base+dx*scale[:,None]
        cost,grad_q=quality_cost(pos,cells,initial_det,goal);N=len(cells)
        geom=local_geometry(pos,cells)
        # Geometrically invalid points contribute a finite repair gradient.
        if geom[2].min()<=0. or geom[4].min()<.0005:
            return 10.+cost/N,(grad_q*scale[:,None])[movable]/N
        m,field=elasticity.solve(pos,cells,faces);C=m['compliance_J']
        gc,_,_=gradient(pos,cells,loaded,field['displacement'],geom)
        # Half the loss budget is left as headroom for the smooth penalty model.
        slack=max(0.,1.-loss_budget*.5-C/refC);penalty=1000.
        cost=cost/N+penalty*slack**2-.001*(C/refC-1)
        grad=grad_q/N-(2*penalty*slack+.001)*gc/refC
        q=float(geom[4].min())
        if C>=floor and q>best_q:
            sh=measures(pos,cells)
            if sh['dihedral_degrees']['min']>=seed_shape['dihedral_degrees']['min']:
                try:check_boundary_motion(base,pos,faces)
                except RuntimeError:pass
                else:
                    best=pos.copy();best_q=q;hist.append({'evaluation':evaluation,'min_quality':q,'compliance_J':C,'min_dihedral':sh['dihedral_degrees']['min']})
        return cost,(grad*scale[:,None])[movable]
    opt=minimize(objective,np.zeros(int(movable.sum())),method='L-BFGS-B',jac=True,bounds=[(-.8,.8)]*int(movable.sum()),options={'maxiter':iterations,'maxls':30,'ftol':1e-13,'gtol':1e-8,'maxcor':16})
    if not hist:raise RuntimeError('joint solver found no feasible iterate')
    return best,{'evaluations':evaluation,'iterations':int(opt.nit),'optimizer_message':str(opt.message),'target':goal,'loss_budget':loss_budget,'best_history':hist,'after':measures(best,cells)}


def _collapse_boundary_signature(cells,source,target):
    """Transfer the oriented patch boundary through a compatible edge contraction."""
    from .topology_optimization import _boundary_signature,_orientation
    previous=_boundary_signature(cells)
    if previous is None: return None
    result={}
    for key,sign in previous.items():
        substituted=tuple(target if v==source else v for v in key)
        if len(set(substituted))<3: continue
        canon=tuple(sorted(substituted));direction=sign*_orientation(substituted)
        result[canon]=result.get(canon,0)+direction
    if any(abs(v)>1 for v in result.values()): return None
    return {key:sign for key,sign in result.items() if sign}


def _split_candidate(points,cells,indices,edge,u,quality_floor):
    """Lower bound on compliance gain by relaxing the inserted node's 3 DOFs."""
    from fem_comparison.elasticity import _strain_displacement,_material_matrix,YOUNG,POISSON
    star=cells[indices];mid=points[list(edge)].mean(axis=0);umid=u[list(edge)].mean(axis=0)
    # A new local ID allows small assembly without copying the global mesh.
    ids,loc=np.unique(star,return_inverse=True);local=loc.reshape(star.shape);p=np.vstack((points[ids],mid));field=np.vstack((u[ids],umid));new=len(ids);sub=[]
    for e in edge:
        child=local.copy();child[star==e]=new;sub.append(child)
    sub=np.vstack(sub); v,B=_strain_displacement(p,sub)
    xyz=p[sub]; edge_sum=sum(np.sum((xyz[:,a]-xyz[:,b])**2,axis=1) for a,b in EDGES)
    if np.min(12*np.cbrt(3*v)**2/edge_sum)<quality_floor:return None
    C=_material_matrix(YOUNG,POISSON);locmid=(sub==new).argmax(axis=1)
    bmid=np.stack([B[i,:,3*j:3*j+3] for i,j in enumerate(locmid)])
    K=np.einsum('n,nij,jk,nkl->il',v,bmid.transpose(0,2,1),C,bmid)
    eps=np.einsum('nij,nj->ni',B,field[sub].reshape(-1,12));r=-np.einsum('n,nij,ni->j',v,bmid,eps@C)
    gain=float(r@np.linalg.solve(K,r));return gain,mid


def reallocate_short_edges(vertices,tets,faces,compliance_floor,quality_target=.18,passes=12):
    """Remove one poorly placed free node, and reuse its ID at a new free edge midpoint.

    A boundary contraction is allowed only along the geometric feature containing
    the source node. End faces stay frozen; displaced-boundary triangles remain
    on their original planes. The final extracted boundary and material volume
    are rechecked after every accepted batch. Both available node count and free
    displacement DOFs are preserved; no hidden clamp nodes can be added.
    """
    from .compliance_flips import _energy
    from .topology_optimization import _boundary_signature
    from fem_comparison.cpu_meshes import extract_boundary
    p=vertices.copy();cells=tets.copy();bound=faces.copy();hist=[]
    met,field=elasticity.solve(p,cells,bound);original_vol=_energy(p,cells,field['displacement'])[1].sum()
    for iteration in range(passes):
        u=field['displacement'];oe,ov,oq=_energy(p,cells,u)
        incident={};edge_inc={}
        for i,tet in enumerate(cells):
            for v in tet:incident.setdefault(int(v),[]).append(i)
            for a,b in combinations(tet,2):edge_inc.setdefault(tuple(sorted((int(a),int(b)))),[]).append(i)
        movable,_=free_coordinates(p,bound); count=len(p); collapses=[]
        worst_ids=np.flatnonzero(oq<quality_target); active_ids=np.unique(cells[worst_ids]);
        for v in active_ids:
            if np.isclose(p[v,1],3.,atol=1.e-12) or not movable[v].any():continue
            ids=incident[v];star=cells[ids];q0=oq[ids].min()
            for neighbor in set(star.ravel())-{v}:
                if not np.array_equal(p[v,~movable[v]],p[neighbor,~movable[v]]):continue
                kept=star[~np.any(star==neighbor,axis=1)].copy(); kept[kept==v]=neighbor
                if len(kept)==0:continue
                x=p[kept]; det=np.linalg.det(x[:,1:]-x[:,:1]);
                if det.min()<=1e-12 or not np.isclose(det.sum()/6,ov[ids].sum(),rtol=1e-10,atol=1e-13):continue
                E,V,Q=_energy(p,kept,u); better=Q.min()-q0
                if better<max(.01,.25*q0):continue
                if _collapse_boundary_signature(star,v,neighbor)!=_boundary_signature(kept): continue
                collapses.append((float(q0),float(E.sum()-oe[ids].sum()),v,ids,kept))
        if not collapses:break
        # Rank split candidates cheaply using the local strain energy per star.
        # Use existing edges only away from loaded/clamped ends and avoid any
        # newly poor split children. The final ordering uses computable gains.
        births=[];candidates=[]
        for edge,ids in edge_inc.items():
            mid=p[list(edge)].mean(axis=0)
            if np.isclose(mid[1],3.,atol=1e-12) or oq[ids].min()<quality_target:continue
            candidates.append((float(oe[ids].sum()),edge,ids))
        for _,edge,ids in sorted(candidates,reverse=True)[:600]:
            cand=_split_candidate(p,cells,ids,edge,u,quality_target)
            if cand:
                gain,mid=cand;births.append((gain,edge,ids,mid,set(cells[ids].ravel())))
        births.sort(key=lambda x:-x[0]);used=set();removed=set();addition=[];predicted=0.;updates=[]
        for q0,loss,v,ids,kept in sorted(collapses,key=lambda x:(x[0],x[1])):
            collapse_set=set(cells[ids].ravel())
            if used.intersection(collapse_set):continue
            for gain,edge,split_ids,mid,birth_set in births:
                if met['compliance_J']+predicted-loss+gain<compliance_floor:break
                if used.intersection(birth_set) or collapse_set.intersection(birth_set):continue
                star=cells[split_ids];addition.append(kept)
                for end in edge:
                    child=star.copy();child[star==end]=v;addition.append(child)
                used.update(collapse_set|birth_set);removed.update(ids);removed.update(split_ids)
                predicted+=gain-loss;updates.append((int(v),mid));break
        if not updates:break
        newcells=np.vstack([np.delete(cells,sorted(removed),axis=0),*addition]);newp=p.copy()
        for v,mid in updates:newp[v]=mid
        bf=extract_boundary(newcells);m2,s2=elasticity.solve(newp,newcells,bf)
        det=np.linalg.det(newp[newcells[:,1:]]-newp[newcells[:,:1]])
        if np.min(det)<=0 or not np.isclose(det.sum()/6,original_vol,rtol=1e-11) or len(np.unique(newcells))!=count:raise RuntimeError('failed split/collapse volume or active vertex count checks')
        if m2['free_dofs']!=met['free_dofs'] or m2['compliance_J']+1e-7<max(compliance_floor,met['compliance_J']+predicted):raise RuntimeError('failed paired refinement variational bound')
        p,cells,bound,met,field=newp,newcells,bf,m2,s2
        hist.append({'pass':iteration,'relocated_ids':[i for i,_ in updates],'predicted_gain_J':predicted,'compliance_J':met['compliance_J'],'quality':measures(p,cells)})
    return p,cells,bound,hist
