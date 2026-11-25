"""NumPy reference implementation of batched tetrahedron SAT with bounded scratch."""
import numpy as np
from .geometry import _EDGE_IDS, _FACE_IDS, _NORMAL_EPS


def _nonseparated(a,b,axes,usable,tol):
    lengths=np.linalg.norm(axes,axis=2)
    unit=np.divide(axes,lengths[:,:,None],out=np.zeros_like(axes),where=usable[:,:,None])
    pa=a@unit.swapaxes(1,2)
    pb=b@unit.swapaxes(1,2)
    depth=np.minimum(pa.max(1),pb.max(1))-np.maximum(pa.min(1),pb.min(1))
    return ~np.any(usable & (depth<=tol),axis=1)


def _overlap(a,b,tol):
    origin=a[:,:1]
    a,b=a-origin,b-origin
    def nondegenerate(c):
        x,y,z=(c[:,i]-c[:,0] for i in (1,2,3))
        return np.einsum('ij,ij->i',x,np.cross(y,z))!=0.
    faces=np.concatenate((a[:,_FACE_IDS],b[:,_FACE_IDS]),axis=1)
    axes=np.cross(faces[:,:,1]-faces[:,:,0],faces[:,:,2]-faces[:,:,0])
    usable=np.linalg.norm(axes,axis=2)>0.
    result=nondegenerate(a)&nondegenerate(b)&usable.all(1)&_nonseparated(a,b,axes,usable,tol)
    rem=np.flatnonzero(result)
    if len(rem):
        ea=a[rem][:,_EDGE_IDS[:,1]]-a[rem][:,_EDGE_IDS[:,0]]
        eb=b[rem][:,_EDGE_IDS[:,1]]-b[rem][:,_EDGE_IDS[:,0]]
        cross=np.cross(ea[:,:,None],eb[:,None,:]).reshape(-1,36,3)
        min_norm=(_NORMAL_EPS*np.linalg.norm(ea,axis=2)[:,:,None]*np.linalg.norm(eb,axis=2)[:,None,:]).reshape(-1,36)
        result[rem]=_nonseparated(a[rem],b[rem],cross,np.linalg.norm(cross,axis=2)>min_norm,tol)
    return result


def overlap_any(proposals,accepted,tol):
    result=np.zeros(len(proposals),bool)
    if not len(proposals) or not len(accepted):return result
    low,high=proposals.min(1),proposals.max(1)
    old_low,old_high=accepted.min(1),accepted.max(1)
    for j in range(0,len(accepted),64):
        for i in range(0,len(proposals),256):
            ids=i+np.flatnonzero(~result[i:i+256])
            if not len(ids):continue
            depth=(np.minimum(high[ids,None],old_high[None,j:j+64])
                   -np.maximum(low[ids,None],old_low[None,j:j+64]))
            row,col=np.nonzero(np.all(depth>tol,axis=2))
            for k in range(0,len(row),512):
                ix=ids[row[k:k+512]]
                keep=~result[ix]
                if not keep.any():continue
                ix,last=ix[keep],j+col[k:k+512][keep]
                result[ix[_overlap(proposals[ix],accepted[last],tol)]]=True
    return result
