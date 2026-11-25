"""Redistribute U-bracket nodes using a P1 patch or hierarchical P2 curvature estimate.

The metric integrates variation of all three displacement-gradient components
over the thickness. Actual final comparisons and compliance selection use P1.
"""
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from skfem import Basis,FacetBasis,ElementVector,ElementTetP2,LinearForm,asm,condense,solve
from skfem.models.elasticity import linear_elasticity,lame_parameters

POLYGON=np.array([(0.,0.),(3.,0.),(3.,3.),(2.,3.),(2.,1.),(1.,1.),(1.,3.),(0.,3.)])

def patch_hessian(centres, gradients, volumes, neighbours=32):
    """Recover symmetric displacement Hessians by volume-weighted gradient fits.

    The fit includes a constant to remove bias from one-sided boundary patches.
    Coordinates are scaled by the outer patch radius before the 4x4 solve.
    """
    distance, ids = cKDTree(centres).query(centres, k=min(neighbours,len(centres)))
    radius = distance[:,-1,None]
    if np.any(radius<=0):
        raise ValueError('gradient recovery needs a nonzero patch radius')
    delta = (centres[ids]-centres[:,None,:])/radius[:,:,None]
    X = np.concatenate((np.ones((*distance.shape,1)),delta),axis=2)
    w = volumes[ids]/(.10+(distance/radius)**2)
    normal = np.einsum('nki,nk,nkj->nij',X,w,X)
    observations = gradients[ids].reshape(len(centres),-1,9)
    rhs = np.einsum('nki,nk,nkj->nij',X,w,observations)
    fitted = np.linalg.solve(normal,rhs)[:,1:,:].reshape(-1,3,3,3)
    # Axes: cell, direction of differentiation, displacement, gradient direction.
    hessian = fitted.transpose(0,1,3,2)/radius[:,:,None,None]
    return .5*(hessian+hessian.transpose(0,2,1,3))


def recover_p1_curvature(p,t,f):
    from fem_comparison.elasticity import solve as solve_p1
    metrics, fields = solve_p1(p,t,f)
    xyz=p[t]; jacobian=xyz[:,1:]-xyz[:,:1]
    volumes=np.linalg.det(jacobian)/6
    if np.any(volumes<=0): raise ValueError('positive tetrahedra required')
    displacement=fields['displacement'][t]
    gradients=np.linalg.solve(jacobian,displacement[:,1:]-displacement[:,:1]).transpose(0,2,1)
    centres=xyz.mean(axis=1)
    H=patch_hessian(centres,gradients,volumes)
    # A ZZ-style gradient variation indicator. The final metric uses planar
    # directional derivatives of all three components of the 3D gradient.
    planar=H[:,:,:2,:]
    gram=np.einsum('tkac,tkbc->tab',planar,planar)
    return centres[:,:2],gram,volumes,{'pilot_dofs':metrics['free_dofs'],
        'pilot_mean_loaded_ux_m':metrics['mean_loaded_ux_m'], 'patch_size':32,
        'method':'weighted least-squares recovery of displacement-gradient variation from the 3078-DOF P1 solve'}

def recover_curvature(p,t):
    from fem_comparison.reference import compact_mesh
    mesh=compact_mesh(p,t); element=ElementVector(ElementTetP2()); basis=Basis(mesh,element,intorder=2)
    left=mesh.facets_satisfying(lambda x:np.isclose(x[1],3.)&(x[0]<1.),boundaries_only=True)
    right=mesh.facets_satisfying(lambda x:np.isclose(x[1],3.)&(x[0]>2.),boundaries_only=True)
    stiffness=asm(linear_elasticity(*lame_parameters(200e9,.3)),basis)
    @LinearForm
    def traction(v,w): return 1e6*v[0]
    force=asm(traction,FacetBasis(mesh,element,facets=right,intorder=2))
    u=solve(*condense(stiffness,force,D=basis.get_dofs(facets=left).all()))
    vertices=mesh.p.T; cells=mesh.t.T; node_disp=u[basis.nodal_dofs].T
    edges=mesh.edges.T; e=vertices[edges[:,1]]-vertices[edges[:,0]]
    dx,dy,dz=e.T
    # e^T H e = -8 (u_P2(midpoint)-u_P1_interpolant(midpoint)).
    A=np.column_stack((dx*dx,dy*dy,dz*dz,2*dx*dy,2*dy*dz,2*dz*dx))
    surplus=u[basis.edge_dofs].T - node_disp[edges].mean(axis=1)
    ids=mesh.t2e.T; h=np.linalg.solve(A[ids],-8.*surplus[ids])
    H=h[:,[[0,3,5],[3,1,4],[5,4,2]],:]; planar=H[:,:,:2,:]
    G=np.einsum('tkac,tkbc->tab',planar,planar)
    xyz=vertices[cells];vol=np.abs(np.linalg.det(xyz[:,1:]-xyz[:,:1]))/6
    return xyz.mean(axis=1)[:,:2],G,vol,{'enriched_dofs':int(len(u)-len(basis.get_dofs(facets=left).all())), 'max_edge_surplus_m':float(abs(surplus).max()),'method':'P2-P1 hierarchical edge interpolation defects, reconstruct curvature in 3D; thickness-averaged planar gradient error metric'}


def inside(x):
    return ((x[:,0]<=1.)|(x[:,0]>=2.)|(x[:,1]<=1.))&(x[:,0]>=0)&(x[:,1]>=0)&(x[:,0]<=3)&(x[:,1]<=3)


def _recover_cap_edges(points, cells, boundary_count):
    """Restore missing polygon segments before discarding exterior triangles.

    For this simple polygon, repeatedly flipping crossing diagonals removes an
    obstruction. We only commit convex swaps that strictly reduce the number
    of crossed edges and fail explicitly on an unrecoverable segment.
    """
    def determinant(a,b,c):
        u=points[b]-points[a];v=points[c]-points[a]
        return u[0]*v[1]-u[1]*v[0]
    def crosses(a,b,c,d):
        if len({a,b,c,d})<4: return False
        return (determinant(a,b,c)*determinant(a,b,d)<0.
                and determinant(c,d,a)*determinant(c,d,b)<0.)
    def adjacency():
        faces={}
        for i,tri in enumerate(cells):
            for a,b,c in (tri, np.roll(tri,1), np.roll(tri,2)):
                faces.setdefault(tuple(sorted((a,b))),[]).append((i,c))
        return faces
    fan=adjacency()
    for a in range(boundary_count):
        b=(a+1)%boundary_count
        while tuple(sorted((a,b))) not in fan:
            for (c,d),adj in fan.items():
                if len(adj)!=2 or not crosses(a,b,c,d): continue
                (i,s),(j,r)=adj
                if not crosses(c,d,s,r) or crosses(a,b,s,r): continue
                cells[i],cells[j]=[s,r,c],[r,s,d]
                fan=adjacency()
                break
            else: raise RuntimeError(f'could not recover cap segment {a}--{b}')
    return cells


def weighted_metric(samples,centres,G,vol,exponent,aspect):
    dist,idx=cKDTree(centres).query(samples,k=min(32,len(centres)))
    w=vol[idx]/(dist**2+.025**2);w/=w.sum(axis=1)[:,None]
    gram=np.einsum('nk,nkij->nij',w,G[idx]);e,v=np.linalg.eigh(gram)
    e=np.maximum(e,1e-25);e[:,0]=np.maximum(e[:,0],e[:,1]/aspect**2)
    # Optimal unit-area tensor and density for squared gradient interpolation error in the plane.
    density=e.prod(axis=1)**.25
    density=(density/np.median(density))**exponent;density=np.clip(density,.20,6.)
    shape=(e/np.sqrt(e.prod(axis=1))[:,None])**exponent
    Q=np.einsum('nij,nj,nkj->nik',v,shape,v)
    return density, Q


def adapted_section(centres,G,vol,free_points,*,exponent=.8,aspect=3,iterations=30,resolution=90,seed=3141):
    xy_grid=(np.indices((3*resolution,3*resolution)).reshape(2,-1).T+.5)/resolution
    samples=xy_grid[inside(xy_grid)]
    dens,Q=weighted_metric(samples,centres,G,vol,exponent,aspect)
    kd=cKDTree(samples)
    # Triangular-lattice area per vertex is sqrt(3)/2 times edge length squared.
    scale=np.sqrt((np.sqrt(3)/2)*free_points/(dens.sum()/resolution**2))
    boundary=[]
    for a,b in zip(POLYGON,np.roll(POLYGON,-1,axis=0)):
        s=np.linspace(0.,1.,257); path=a+(b-a)*s[:,None];direction=(b-a)/np.linalg.norm(b-a)
        near=kd.query(path)[1];rate=np.sqrt(dens[near]*np.einsum('i,nij,j->n',direction,Q[near],direction))
        arcl=np.r_[0,np.cumsum(.5*(rate[1:]+rate[:-1])*np.linalg.norm(b-a)/256)]
        count=max(1,int(np.rint(arcl[-1]*scale)))
        boundary.extend(a+(b-a)*np.interp(np.arange(count)/count*arcl[-1],arcl,s)[:,None])
    boundary=np.array(boundary); nb=len(boundary); nfixed=((boundary[:,1]==3)&(boundary[:,0]<=1)).sum()
    ninterior=free_points-(nb-nfixed)
    if ninterior<1: raise ValueError('node budget cannot resolve boundary')
    rng=np.random.default_rng(seed);pick=rng.choice(len(samples),ninterior,replace=False,p=dens/dens.sum());points=np.vstack((boundary,samples[pick]))
    # Lloyd centroidal iteration in the estimated anisotropic metric. The cap boundary stays fixed.
    W=Q*dens[:,None,None]**2;rhs=np.einsum('nij,nj->ni',W,samples)
    for _ in range(iterations):
        near=cKDTree(points).query(samples,k=7)[1];delta=points[near]-samples[:,None]
        dist=np.einsum('nki,nij,nkj->nk',delta,Q,delta)
        owner=near[np.arange(len(samples)),dist.argmin(axis=1)];n=len(points)
        mat=np.stack([np.bincount(owner,weights=W[:,i,j],minlength=n) for i,j in [(0,0),(0,1),(1,0),(1,1)]],axis=1).reshape(-1,2,2)
        vec=np.stack([np.bincount(owner,weights=rhs[:,i],minlength=n) for i in (0,1)],axis=1)
        active=(np.arange(n)>=nb)&(np.linalg.det(mat)>1e-16);trial=np.linalg.solve(mat[active],vec[active,:,None])[:,:,0]
        step=points.copy();step[active]=trial; update=active&inside(step);points[update]=.20*points[update]+.80*step[update]
    cap=_recover_cap_edges(points,Delaunay(points).simplices.copy(),nb)
    cap=cap[inside(points[cap].mean(axis=1))]
    signed=np.cross(points[cap[:,1]]-points[cap[:,0]],points[cap[:,2]]-points[cap[:,0]])
    cap[signed<0]=cap[signed<0][:,[0,2,1]]
    edges=np.sort(cap[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1);keys,counts=np.unique(edges,axis=0,return_counts=True)
    expected={tuple(sorted((i,(i+1)%nb))) for i in range(nb)}
    if {tuple(x) for x in keys[counts==1]}!=expected or not np.isclose(abs(signed).sum()/2,7.):
        raise RuntimeError('cap triangulation did not recover original boundary')
    return points,cap,nb


def extrude_section(xy,cap,nb,nz,backend='cpu',*,mode='batched'):
    from saft3d.backend import create_backend
    from saft3d.thickness_front import _advance_cap
    from saft3d.batched_prism import advance_batched_prisms
    from fem_comparison.cpu_meshes import extract_boundary
    xy=np.ascontiguousarray(xy,dtype=np.float64)
    cap=np.ascontiguousarray(cap,dtype=np.int64)
    p=np.vstack([np.column_stack((xy,np.full(len(xy),z))) for z in np.linspace(-.5,.5,nz)])
    layers=np.arange(len(p),dtype=np.int64).reshape(nz,len(xy));engine=create_backend(backend)
    if mode=='batched':
        cells, evidence = advance_batched_prisms(p,cap,layers,engine)
        launches = evidence['kernel_calls']['batch_front']
    elif mode=='sweep':
        cells=_advance_cap(xy,cap,layers,engine)
        if engine.name=='cuda':
            cells=engine.cp.asnumpy(cells)
        launches=nz-1 if engine.name=='cuda' else 0
    else:
        raise ValueError('mode must be batched or sweep')
    return p,cells,extract_boundary(cells),{'backend':engine.name,'device':engine.device_info,'cap_triangles':len(cap),'cap_vertices':len(xy),'cap_boundary_nodes':nb,'z_planes':nz,'cuda_advance_kernels':launches, 'method':mode+' prism generation on an error-adapted cross-section','cpu_completion':False}



def improve_prism_diagonals(vertices,seed_cells,cap,planes,steps=12):
    """Optimize conforming prism diagonals using total potential at the current P1 solution.

    Every option uses the same six vertices and unchanged horizontal cap. Shared
    sidewall diagonals are single binary variables, so incompatible swaps are rejected.
    """
    from itertools import permutations
    from fem_comparison.elasticity import _strain_displacement, _material_matrix, YOUNG, POISSON, TRACTION, solve
    from fem_comparison.cpu_meshes import extract_boundary
    p=vertices; n=len(p)//planes; cap=np.sort(cap,axis=1);m=len(cap)
    order=np.array(list(permutations(range(3))))
    face_slots=np.array([(0,1,2),(0,1,3),(0,2,3),(1,2,3)])
    lower=np.tile(cap,(planes-1,1)) + np.repeat(np.arange(planes-1),m)[:,None]*n
    candidates=[]; codes=[]
    for perm in order:
        a,b,c=lower[:,perm].T
        pts=np.stack((np.stack((a,b,c,c+n),axis=1),np.stack((a,b,c+n,b+n),axis=1),np.stack((a,a+n,b+n,c+n),axis=1)),axis=1)
        xyz=p[pts];inv=np.linalg.det(xyz[:,:,1:]-xyz[:,:,:1])<0;pts[inv]=pts[inv][:,[1,0,2,3]]
        candidates.append(pts)
        rank=np.argsort(perm);codes.append(sum((rank[j]<rank[i])*(1<<k) for k,(i,j) in enumerate([(0,1),(1,2),(0,2)])))
    opts=np.stack(candidates,axis=1);Nc=len(opts);coords=opts.reshape(-1,4)
    volumes,B=_strain_displacement(p,coords); mat=_material_matrix(YOUNG,POISSON)
    score_code=np.full(8,-1,int);score_code[codes]=np.arange(6)
    # Nodal work for constant traction changes if a loaded face is retriangulated.
    wt=np.zeros(coords.shape)
    for fs in face_slots:
        tri=p[coords[:,fs]];is_load=(np.isclose(tri[:,:,1],3.)&(tri[:,:,0]>=2-1e-10)).all(axis=1)
        weights=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/6*TRACTION
        wt[:,fs]+=np.where(is_load,weights,0)[:,None]
    edge_map={}
    for i,base in enumerate(lower):
        for k,(a,b) in enumerate([(0,1),(1,2),(0,2)]):edge_map.setdefault((base[a],base[b]),[]).append((i,1<<k))
    edges=list(edge_map.values()); code=np.zeros(Nc,int)
    history=[]; tess=seed_cells
    for step in range(steps):
        q,fields=solve(p,tess,extract_boundary(tess)); u=fields['displacement'][coords]
        eps=np.einsum('tij,tj->ti',B,u.reshape(-1,12));energy=np.einsum('ti,ij,tj->t',eps,mat,eps)*volumes
        work=(2*wt*u[:,:,0]).sum(axis=1); scores=(work-energy).reshape(Nc,6,3).sum(axis=2)
        moves=0
        for sweep in range(20):
            nb=0
            for fan in edges:
                gain=0.
                for k,bit in fan:
                    opt=score_code[code[k]^bit]
                    if opt<0:gain=-np.inf;break
                    gain+=scores[k,opt]-scores[k,score_code[code[k]]]
                if gain>1e-11:
                    for k,bit in fan:code[k]^=bit
                    nb+=1
            moves+=nb
            if nb==0:break
        history.append({'ux_mm':q['mean_loaded_ux_m']*1e3,'diagonal_moves':moves})
        tess=opts[np.arange(Nc),score_code[code]].reshape(-1,4)
        if moves==0:break
    q,_=solve(p,tess,extract_boundary(tess));history.append({'ux_mm':q['mean_loaded_ux_m']*1e3,'diagonal_moves':0})
    if np.min(np.diff([h['ux_mm'] for h in history]))<-1e-9:raise RuntimeError('variational check failed')
    return tess,extract_boundary(tess),history
