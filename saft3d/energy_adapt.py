"""Compliance-driven refinement from a computed displacement P1 solution.

Test the equilibrium residual against the three vector-valued nodal hats that
would appear if an edge star were bisected. The old solution is interpolated
exactly in the split space. Relaxing the new node alone gains r.Kee^-1.r in
compliance. For the U load, compliance is exactly proportional to the target
mean ux. No reference displacement is used for marking.
"""
from itertools import combinations
import heapq
import numpy as np
from scipy.spatial import cKDTree

from .bending_refinement import _Cells, fixed_end
from fem_comparison.elasticity import YOUNG, POISSON, TRACTION, solve


EDGES = tuple(combinations(range(4), 2))


def edge_indicators(points, cells, surface, displacement):
    xyz = points[cells]
    mat = xyz[:, 1:] - xyz[:, :1]
    volume = np.linalg.det(mat) / 6
    if np.min(volume) <= 0:
        raise ValueError("edge estimator requires positively oriented tetrahedra")
    g = np.linalg.inv(mat).transpose(0,2,1)
    g = np.concatenate([-g.sum(axis=1,keepdims=True), g],axis=1)
    du = np.einsum("tnd,tni->tdi", displacement[cells], g)
    mu = YOUNG/(2*(1+POISSON))
    lam = 2*mu*POISSON/(1-2*POISSON)
    stress = mu*(du+du.transpose(0,2,1)) + lam*np.trace(du,axis1=1,axis2=2)[:,None,None]*np.eye(3)
    # K_ee for the vector hat. On half of a parent its gradient is 2*grad N_i,
    # on the other half it is 2*grad N_j.
    k = 2*volume[:,None,None,None]*(
        mu*np.sum(g*g,axis=2)[:,:,None,None]*np.eye(3)
        +(lam+mu)*np.einsum("tni,tnj->tnij",g,g))
    forces = volume[:,None,None]*np.einsum("tij,tnj->tni",stress,g)
    pair = np.array(EDGES)
    edges, inverse = np.unique(np.sort(cells[:,pair],axis=2).reshape(-1,2),axis=0,return_inverse=True)
    inverse=inverse.reshape(-1,6)
    kee=np.zeros((len(edges),3,3)); r=np.zeros((len(edges),3))
    # Penalize splits producing very poor shapes, even if the indicator is large.
    min_child_quality=np.full(len(edges),np.inf)
    for e,(i,j) in enumerate(EDGES):
        ix = inverse[:,e]
        np.add.at(kee,ix,k[:,i]+k[:,j])
        np.add.at(r,ix,-forces[:,i]-forces[:,j])
        midpoint=(xyz[:,i]+xyz[:,j])*.5
        for slot in (i,j):
            child=xyz.copy(); child[:,slot]=midpoint
            length_sum=sum(np.sum((child[:,a]-child[:,b])**2,axis=1) for a,b in EDGES)
            np.minimum.at(min_child_quality,ix,12*np.power(1.5*volume,2/3)/length_sum)
    # Constant face traction: the scalar midpoint hat integrates to area/3.
    loaded = surface[np.all(np.isclose(points[surface,1],3.,rtol=0,atol=1e-10),axis=1)
                     & np.all(points[surface,0]>=2.-1e-10,axis=1)]
    indices={tuple(a):i for i,a in enumerate(edges)}
    for tri in loaded:
        x=points[tri]
        area=np.linalg.norm(np.cross(x[1]-x[0],x[2]-x[0]))/2
        for edge in combinations(sorted(tri),2):
            r[indices[edge],0]+=TRACTION*area/3
    correction=np.linalg.solve(kee,r[...,None])[...,0]
    gain=np.einsum('ni,ni->n',r,correction)
    return edges,gain,min_child_quality,inverse


def adaptive_fit(points,cells,faces,free_dofs=3078,*,quality_scale=.25,batch_fraction=.18,
                 clamp_nodes=None):
    """Build a conforming nested mesh with exactly free_dofs final P1 unknowns.

    Stars in a single batch share no tetrahedron. The three corrections on
    each accepted edge then have mutually disjoint volume support. Fixed
    clamp nodes have zero gain and are not added. The independent solve after
    each batch should improve the compliance by at least the predicted sum.
    """
    if free_dofs % 3: raise ValueError("free displacement budget must be a multiple of three")
    used=np.unique(cells); p=list(np.asarray(points)[used])
    mesh=_Cells(np.searchsorted(used,cells)); shell=_Cells(np.searchsorted(used,faces))
    goal=free_dofs//3; free=len(p)-np.count_nonzero(fixed_end(p))
    if free>goal:raise ValueError("source already exceeds target")
    history=[]
    while True:
        # Pre-allocate at most the original reference-free budget of fixed
        # boundary nodes. Refinement of a Dirichlet face creates no DOF by
        # itself but gives nearby cells more possible future bisections.
        if clamp_nodes is not None:
            allowed=max(4,round(clamp_nodes*(free/goal)**(2/3)))
            while len(p)-free < allowed:
                xp=np.asarray(p); fixed=fixed_end(xp); edges=np.array(list(shell.incident))
                edges=edges[np.all(fixed[edges],axis=1)]
                length=np.linalg.norm(xp[edges[:,1]]-xp[edges[:,0]],axis=1)
                edge=tuple(edges[length.argmax()]);i=len(p);p.append((p[edge[0]]+p[edge[1]])*.5)
                mesh.split(edge,i);shell.split(edge,i)
        xyz=np.array(p); cells=np.asarray(list(mesh.cells.values()),int); boundary=np.asarray(list(shell.cells.values()),int)
        metrics, fields = solve(xyz,cells,boundary)
        compliance=metrics['compliance_J']
        if history and compliance < history[-1]['compliance_J']+history[-1]['predicted_gain_J']-1e-7*compliance:
            raise ArithmeticError("observed compliance failed its predicted enrichment lower bound")
        row={'nodes':len(p),'free_dofs':int(3*free),'tets':len(cells),
             'compliance_J':compliance,'mean_loaded_ux_m':metrics['mean_loaded_ux_m']}
        history.append(row)
        if free==goal:break
        edges,gain,q,local_ids=edge_indicators(xyz,cells,boundary,fields['displacement'])
        free_edge=~fixed_end(xyz[edges].mean(axis=1))
        scores=gain*(np.minimum(1.,q/quality_scale)**2 if quality_scale>0 else 1)
        scores[~free_edge]=-np.inf
        order=np.argsort(-scores,kind='stable')
        patch_faces=[[] for _ in edges]
        for tc,ids in enumerate(local_ids):
            for edge_id in ids:patch_faces[edge_id].append(tc)
        marked=[];occupied=set()
        batch=min(goal-free,max(2,int(batch_fraction*free)))
        for e in order:
            if not np.isfinite(scores[e]):break
            if not occupied.isdisjoint(patch_faces[e]):continue
            marked.append(e);occupied.update(patch_faces[e])
            if len(marked)>=batch:break
        if not marked:raise RuntimeError("no eligible free edges")
        row['marked_edges']=len(marked);row['predicted_gain_J']=float(gain[marked].sum())
        for e in marked:
            edge=tuple(edges[e]); mid=len(p);p.append((p[edge[0]]+p[edge[1]])*.5)
            mesh.split(edge,mid);shell.split(edge,mid)
        free+=len(marked)
    return xyz,cells,boundary,history


def enriched_strain_metric(points,cells,aspect_limit=5.,neighbors=9):
    """Estimate strain-gradient energy from a P2 pilot, not a reference solution.

    Hessians are recovered exactly from the six directional midpoint surpluses
    of each quadratic. The returned metric equidistributes the leading local
    H1 interpolation-error model at fixed complexity (3D).
    """
    from skfem import Basis, FacetBasis, ElementTetP2, ElementVector, LinearForm, asm, condense
    from skfem import solve as fe_solve
    from skfem.models.elasticity import linear_elasticity, lame_parameters
    from fem_comparison.reference import compact_mesh
    from fem_comparison.elasticity import _material_matrix
    mesh=compact_mesh(points,cells)
    elem=ElementVector(ElementTetP2());basis=Basis(mesh,elem,intorder=2)
    left=mesh.facets_satisfying(lambda x: np.isclose(x[1],3)&(x[0]<1),boundaries_only=True)
    right=mesh.facets_satisfying(lambda x: np.isclose(x[1],3)&(x[0]>2),boundaries_only=True)
    @LinearForm
    def traction(test,w): return TRACTION*test[0]
    load=asm(traction,FacetBasis(mesh,elem,facets=right,intorder=2))
    stiff=asm(linear_elasticity(*lame_parameters(YOUNG,POISSON)),basis)
    u=fe_solve(*condense(stiff,load,D=basis.get_dofs(facets=left).all()))
    edges=mesh.edges.T;xyz=mesh.p.T;it=mesh.t2e.T
    surplus=u[basis.edge_dofs].T-u[basis.nodal_dofs].T[edges].mean(axis=1)
    e=xyz[edges[:,1]]-xyz[edges[:,0]]
    x,y,z=e.T
    coeff=np.stack((x*x,y*y,z*z,2*x*y,2*y*z,2*z*x),axis=1)
    # directional second derivative times squared edge length
    derivatives=np.linalg.solve(coeff[it],-8*surplus[it])
    h=np.empty((len(it),3,3,3))
    for i,(a,b) in enumerate(((0,0),(1,1),(2,2),(0,1),(1,2),(2,0))):
        h[:,:,a,b]=derivatives[:,i]; h[:,:,b,a]=derivatives[:,i]
    strain_grad=np.stack((h[:,0,0,:],h[:,1,1,:],h[:,2,2,:],
                          h[:,0,1,:]+h[:,1,0,:],h[:,1,2,:]+h[:,2,1,:],
                          h[:,0,2,:]+h[:,2,0,:]),axis=1)
    tensor=strain_grad.transpose(0,2,1)@_material_matrix(YOUNG,POISSON)@strain_grad
    eig,vec=np.linalg.eigh(tensor)
    # Prevent unbounded element aspect and smooth isolated almost-affine zones.
    floor=max(1.e-14,np.median(eig[:,-1])*0.012)
    eig=np.maximum(eig,np.maximum(floor,eig[:,-1]/aspect_limit**2)[:,None])
    eig*=np.prod(eig,axis=1)[:,None]**(-.2)
    tensor=(vec*eig[:,None,:])@vec.transpose(0,2,1)
    tree=cKDTree(xyz[mesh.t.T].mean(axis=1))
    def metric(location):
        dist,ids=tree.query(location,k=min(neighbors,len(it)))
        weight=1/(np.atleast_1d(dist)**2+1.e-5)
        return np.average(tensor[np.atleast_1d(ids)],weights=weight,axis=0)
    return metric,{'estimator':'P2 hierarchical midpoint surplus -> strain-gradient H1 metric',
                   'pilot_tets':len(it),'pilot_free_dofs':int(basis.N-len(basis.get_dofs(facets=left).all())),
                   'aspect_limit':aspect_limit,'neighbors':neighbors,'relative_tensor_floor':.012}


def metric_fit(points,cells,faces,metric,free_dofs):
    used=np.unique(cells); p=list(np.asarray(points)[used])
    mesh=_Cells(np.searchsorted(used,cells)); shell=_Cells(np.searchsorted(used,faces))
    free=len(p)-int(np.count_nonzero(fixed_end(p))); budget=free_dofs//3
    if free_dofs%3 or free>budget:raise ValueError('invalid P1 budget')
    queue=[]
    def add(edge):
        a,b=p[edge[0]],p[edge[1]]
        length=b-a;value=float(length@metric(.5*(a+b))@length)
        heapq.heappush(queue,(-value,edge))
    for edge in mesh.incident:add(edge)
    while free<budget:
        if not queue:raise RuntimeError('no edge passes the shape safeguard')
        _,edge=heapq.heappop(queue)
        if edge not in mesh.incident:continue
        midpoint=.5*(p[edge[0]]+p[edge[1]])
        # Spatially varying metrics need not rank nested physical edges
        # monotonically. Prevent repeated subdivisions collapsing an altitude.
        star=np.array([mesh.cells[i] for i in mesh.incident[edge]],dtype=int)
        x=np.array([[p[j] for j in tet] for tet in star])
        v=np.linalg.det(x[:,1:]-x[:,:1])/6
        if np.any(v<2e-10):continue
        quality=lambda c,w:12*(3*w)**(2/3)/sum(np.sum((c[:,a]-c[:,b])**2,axis=1) for a,b in EDGES)
        base_quality=quality(x,v)
        good=True
        for endpoint in edge:
            child=x.copy();child[star==endpoint]=midpoint
            if np.any(quality(child,v*.5)<np.minimum(.14,base_quality*.99)):
                good=False;break
        if not good:continue
        i=len(p);p.append(midpoint);free+=int(not fixed_end(midpoint))
        neighbors=mesh.split(edge,i);shell.split(edge,i)
        for j in sorted(neighbors):add((j,i))
    return np.asarray(p),np.asarray(list(mesh.cells.values())),np.asarray(list(shell.cells.values()))
