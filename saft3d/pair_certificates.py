"""Batched sufficient separation certificates for the independent mesh auditor.

Only pairs that pass a strict, vertex-by-vertex plane check are removed.
Ambiguous pairs remain eligible for the scalar certificate and LP audit.
"""
from itertools import combinations
import numpy as np

EDGES = np.array(list(combinations(range(4),2)))


def broad_pairs(low, high, excluded, tolerance, chunk=2048):
    """Sweep AABBs along a spatial axis, returning bounded batches of pairs."""
    extent = high.max(axis=0)-low.min(axis=0)
    axis = int(np.argmax(extent/np.maximum(np.mean(high-low,axis=0),1e-30)))
    order = np.argsort(low[:,axis],kind='stable')
    starts = low[order,axis]
    limits = np.searchsorted(starts,high[order,axis]+tolerance,side='right')
    batches = []
    count = 0
    for position,i in enumerate(order):
        if excluded[i]: continue
        ids=order[position+1:limits[position]]
        hits=(~excluded[ids] & np.all(low[ids]<=high[i]+tolerance,axis=1)
              & np.all(low[i]<=high[ids]+tolerance,axis=1))
        ids=ids[hits]
        if not len(ids): continue
        batches.append(np.column_stack((np.minimum(i,ids),np.maximum(i,ids))))
        count+=len(ids)
        if count>=chunk:
            data=np.concatenate(batches)
            for first in range(0,len(data),chunk): yield data[first:first+chunk]
            batches=[];count=0
    if batches:yield np.concatenate(batches)


def _units(axes, eps):
    length=np.linalg.norm(axes,axis=-1,keepdims=True)
    return np.divide(axes,length,out=np.zeros_like(axes),where=length>eps)


def _strict_gap(a,b,axes,margin,eps):
    directions=_units(axes,eps).swapaxes(1,2)
    pa,pb=a@directions,b@directions
    gaps=np.maximum(pb.min(axis=1)-pa.max(axis=1),pa.min(axis=1)-pb.max(axis=1))
    return np.any(gaps>margin,axis=1)


def certified(tets, coords, pairs, face_index, margin, roundoff):
    """True iff a plane was verified (arrays reference the same global mesh)."""
    left,right=pairs.T
    a,b=coords[left],coords[right]
    matches=tets[left,:,None]==tets[right,None,:]
    ma,mb=matches.any(axis=2),matches.any(axis=1)
    together=ma.sum(axis=1)
    scale=np.max(np.maximum(a.max(1),b.max(1))-np.minimum(a.min(1),b.min(1)),axis=1)
    result=np.zeros(len(pairs),dtype=bool)
    for nshared in (0,1,2,3):
        idx=np.flatnonzero((together==nshared)&(scale>0))
        if not len(idx):continue
        aa,bb=a[idx],b[idx]
        am,bm=ma[idx],mb[idx]
        origin=aa[np.arange(len(idx)),am.argmax(1)] if nshared else aa[:,0]
        aa=(aa-origin[:,None])/scale[idx,None,None]
        bb=(bb-origin[:,None])/scale[idx,None,None]
        if nshared==0:
            tri=np.concatenate((aa[:,face_index],bb[:,face_index]),axis=1)
            axes=np.cross(tri[:,:,1]-tri[:,:,0],tri[:,:,2]-tri[:,:,0])
            ok=_strict_gap(aa,bb,axes,margin,roundoff)
            rem=np.flatnonzero(~ok)
            if len(rem):
                # An edge cross-product can also supply a valid separating plane.
                ea=aa[rem][:,EDGES[:,1]]-aa[rem][:,EDGES[:,0]]
                eb=bb[rem][:,EDGES[:,1]]-bb[rem][:,EDGES[:,0]]
                axes=np.cross(ea[:,:,None],eb[:,None,:]).reshape(len(rem),36,3)
                ok[rem]=_strict_gap(aa[rem],bb[rem],axes,margin,roundoff)
            result[idx]=ok
            continue
        common=aa[am].reshape(len(idx),nshared,3)
        exclusive_a=aa[~am].reshape(len(idx),4-nshared,3)
        exclusive_b=bb[~bm].reshape(len(idx),4-nshared,3)
        def verify(normal):
            unit=_units(normal,roundoff)[:,:,None]
            plane_ok=np.max(abs(common@unit),axis=1)[:,0]<=roundoff
            da=(exclusive_a@unit)[:,:,0]; db=(exclusive_b@unit)[:,:,0]
            opposite=((da.max(1)<-margin)&(db.min(1)>margin))|((db.max(1)<-margin)&(da.min(1)>margin))
            return plane_ok & opposite
        if nshared==3:
            result[idx]=verify(np.cross(common[:,1],common[:,2]))
            continue
        directions=np.concatenate((-exclusive_a,exclusive_b),axis=1)
        if nshared==2:
            edge=_units(common[:,1],roundoff)
            directions-=np.sum(directions*edge[:,None],axis=2)[:,:,None]*edge[:,None]
        directions=_units(directions,roundoff)
        ok=verify(directions.sum(axis=1))
        if not ok.all():
            if nshared==2:rays=np.cross(edge[:,None],directions)
            else:
                pair=np.asarray(list(combinations(range(6),2)))
                rays=np.cross(directions[:,pair[:,0]],directions[:,pair[:,1]])
            rays=_units(rays,roundoff)
            rays=np.concatenate((rays,-rays),axis=1)
            feasible=(directions@rays.swapaxes(1,2)).min(axis=1)>=-roundoff
            ok|=verify((rays*feasible[:,:,None]).sum(axis=1))
        result[idx]=ok
    return result
