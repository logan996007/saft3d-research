"""Move nodes within the U-bracket domain to reduce compliance discretization error.

For uniform end traction the measured displacement is proportional to f.T @ u.
The conforming Galerkin solution satisfies C_exact-C_h = ||u-u_h||_K**2.
Shape differentiation of C_h therefore needs one current P1 solve and no target
value or higher-order solution. All connectivity and essential DOFs stay fixed.
"""
from itertools import combinations
import numpy as np
from fem_comparison import elasticity

EDGES = list(combinations(range(4),2))


def local_geometry(points,cells):
    x=points[cells]
    edge=x[:,1:]-x[:,:1]
    vol=np.linalg.det(edge)/6
    edge_sum=sum(np.einsum('ij,ij->i',x[:,a]-x[:,b],x[:,a]-x[:,b]) for a,b in EDGES)
    ratio=12*np.cbrt((3*vol)**2)/edge_sum
    return x,edge,vol,edge_sum,ratio


def accumulate(local,indices,node_count):
    ids=indices.ravel()
    return np.column_stack([np.bincount(ids,weights=local[...,k].ravel(),minlength=node_count) for k in range(3)])


def free_coordinates(points, boundary):
    """Positive areas retain each planar boundary patch and its exterior edges."""
    x=points[boundary]
    cross=np.cross(x[:,1]-x[:,0],x[:,2]-x[:,0])
    n=cross/np.linalg.norm(cross,axis=1)[:,None]
    axis=np.argmax(np.abs(n),axis=1)
    if np.any(np.max(np.abs(n),axis=1)<1-1e-10):
        raise ValueError("U-bracket relocation expects axis-aligned planar boundary patches.")
    movable=np.ones(points.shape,dtype=bool)
    for i in range(3):
        movable[boundary[:,i],axis]=False
    end=np.isclose(x[:,:,1],3.).all(axis=1)
    loaded=boundary[end & (x[:,:,0]>=2.-1e-10).all(axis=1)]
    return movable,loaded


def gradient(points,cells,loaded,displacement,geometry=None):
    """Coordinate derivative of C_h, including end traction change for moving nodes."""
    x,edge,vol,edge_sum,quality=geometry if geometry is not None else local_geometry(points,cells)
    dN=np.empty((len(cells),4,3))
    dN[:,1:]=np.linalg.inv(edge).transpose(0,2,1)
    dN[:,0]=-dN[:,1:].sum(axis=1)
    du=displacement[cells].transpose(0,2,1)@dN
    strain=(du+du.transpose(0,2,1))/2
    shear=elasticity.YOUNG/(2*(1+elasticity.POISSON))
    lame=2*shear*elasticity.POISSON/(1-2*elasticity.POISSON)
    stress=2*shear*strain + lame*strain.trace(axis1=1,axis2=2)[:,None,None]*np.eye(3)
    energy=np.einsum('nij,nij->n',stress,strain)
    tensor=2*du.transpose(0,2,1)@stress-energy[:,None,None]*np.eye(3)
    local=vol[:,None,None]*(dN@tensor.transpose(0,2,1))
    grad=accumulate(local,cells,len(points))
    if len(loaded):
        # 2 f' u, with f assembled by exact integration on the moving trace.
        a,b,c=points[loaded].transpose(1,0,2)
        n=np.cross(b-a,c-a); n/=np.linalg.norm(n,axis=1)[:,None]
        gb=.5*np.cross(c-a,n); gc=.5*np.cross(n,b-a)
        d_area=np.stack([-gb-gc,gb,gc],axis=1)
        ux=displacement[loaded,0].mean(axis=1)
        grad+=accumulate((2*elasticity.TRACTION*ux)[:,None,None]*d_area,loaded,len(points))
    return grad,energy*vol,dN


def barrier(s):
    active=s<1
    val=np.zeros_like(s); coeff=np.zeros_like(s)
    val[active]=s[active]-1-np.log(s[active])
    coeff[active]=s[active]-1 # d/d log(s)
    return val,coeff


def check_boundary_motion(original, moved, boundary):
    """Check each original plane, face winding and per-plane area after relocation."""
    movable,_=free_coordinates(original,boundary)
    if not np.array_equal(original[~movable],moved[~movable]):
        raise RuntimeError("A geometrically constrained coordinate was changed")
    before=original[boundary]
    after=moved[boundary]
    n0=np.cross(before[:,1]-before[:,0],before[:,2]-before[:,0])
    n1=np.cross(after[:,1]-after[:,0],after[:,2]-after[:,0])
    if np.any(np.einsum('ij,ij->i',n0,n1)<=0):
        raise RuntimeError("A boundary face lost its positive oriented area")
    axis=np.argmax(np.abs(n0),axis=1)
    areas0=np.linalg.norm(n0,axis=1)*.5
    areas1=np.linalg.norm(n1,axis=1)*.5
    # Coplanar, separately oriented material faces belong to the same plane.
    key=np.column_stack((axis, np.sign(n0[np.arange(len(n0)),axis]),
                         before[np.arange(len(n0)),0,axis]))
    _,classes=np.unique(key,axis=0,return_inverse=True)
    a=np.bincount(classes,weights=areas0); b=np.bincount(classes,weights=areas1)
    relative=float(np.max(np.abs(a-b)/a))
    if relative>1e-10: raise RuntimeError("A boundary plane's area changed")
    return {'boundary_coordinate_constraints_exact':True,'all_face_orientations_preserved':True,
            'max_plane_area_relative_change':relative,
            'max_vertex_motion_m':float(np.linalg.norm(moved-original,axis=1).max()),
            'mean_vertex_motion_m':float(np.linalg.norm(moved-original,axis=1).mean())}


def relocate(vertices,cells,faces,iterations=100,weight=.004,on_step=None):
    if not isinstance(iterations,(int,np.integer)) or iterations<0: raise ValueError("iterations must be nonnegative")
    if not np.isfinite(weight) or weight<=0: raise ValueError("barrier weight must be positive")
    points=np.array(vertices,dtype=float,copy=True)
    movable,loaded=free_coordinates(points,faces)
    geom=local_geometry(points,cells)
    vol0=geom[2].copy()
    if np.min(vol0)<=0: raise ValueError("Input tetrahedra must be positive.")
    initial,fields=elasticity.solve(points,cells,faces)
    C0=initial['compliance_J']
    _,energies,_=gradient(points,cells,loaded,fields['displacement'],geom)
    q_limit=min(.012,.5*np.min(geom[4]))
    vmask=movable.ravel()
    # Scale changes of variables by a local compliance-weighted cell stiffness.
    stiff=energies/(geom[3]/6)
    wnodes=np.bincount(cells.ravel(),weights=np.repeat(stiff,4),minlength=len(points))
    typical=np.median(wnodes[wnodes>0])
    scale=np.sqrt(typical/np.maximum(wnodes,typical*.02)).clip(.05,5.)
    memory=[]

    def evaluate(p, cached=None):
        geo=local_geometry(p,cells)
        x,edge,vol,edge_sum,q=geo
        if np.min(vol/vol0)<.009 or q.min()<q_limit: return None
        met,sol=cached if cached else elasticity.solve(p,cells,faces)
        gc,_,dN=gradient(p,cells,loaded,sol['displacement'],geo)
        # The barrier is singular at the admissibility floor rather than at
        # zero, otherwise one cell reaching that floor can stall all nodes.
        vrel=vol/vol0
        bv,cv=barrier((vrel-.009)/(.30-.009))
        cv*=vrel/(vrel-.009)
        bq,cq=barrier((q-q_limit)/(.20-q_limit))
        cq*=q/(q-q_limit)
        dq=(2/3)*dN-8*(x-x.mean(axis=1)[:,None,:])/edge_sum[:,None,None]
        gb=accumulate(cv[:,None,None]*dN+cq[:,None,None]*dq,cells,len(p))
        cost=-met['compliance_J']/C0 + weight*(bv+bq).mean()
        grad=(-gc/C0 + weight*gb/len(cells))*scale[:,None]
        return cost,grad.ravel()[vmask],met,sol,q.min()

    current=evaluate(points,(initial,fields))
    history=[]
    def record(i):
        cost,g,met,_,qmin=current
        history.append({'iteration':i,'mean_loaded_ux_m':met['mean_loaded_ux_m'],
                        'min_mean_ratio':float(qmin),'objective':float(cost),'gradient_norm':float(np.linalg.norm(g))})
        if on_step is not None: on_step(points,history)
    record(0)
    for i in range(iterations):
        cost,g=current[:2]
        z=g.copy(); alpha=[]
        for s,y,rho in reversed(memory):
            a=rho*np.dot(s,z); alpha.append(a); z-=a*y
        if memory:
            s,y,_=memory[-1]
            z*=np.dot(s,y)/np.dot(y,y)
        else: z*=max(.01,.025/np.max(np.abs(z)))
        for (s,y,rho),a in zip(memory,reversed(alpha)):
            z+=s*(a-rho*np.dot(y,z))
        direction=-z; slope=np.dot(g,direction)
        if not slope < 0: memory.clear();direction=-g; slope=-np.dot(g,g)
        delta=np.zeros(points.size);delta[vmask]=direction
        delta=delta.reshape(points.shape)*scale[:,None]
        factor=1.
        for _ in range(28):
            candidate=points+factor*delta
            nxt=evaluate(candidate)
            if nxt is not None and nxt[0]<cost+1e-4*factor*slope: break
            factor*=.5
        else: break
        y=nxt[1]-g; s=factor*direction; sy=s@y
        if sy>1e-12*np.linalg.norm(s)*np.linalg.norm(y):
            memory.append((s,y,1/sy));memory=memory[-16:]
        current=nxt;points=candidate
        record(i+1)
    check_boundary_motion(np.asarray(vertices),points,faces)
    return points,current[2],current[3],history
