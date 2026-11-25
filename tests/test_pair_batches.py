"""Keep chunking and conservative plane certificates equivalent at their boundaries."""
from itertools import combinations
import numpy as np
from saft3d.batch_overlap import overlap_any
from saft3d.geometry import tet_overlap
from saft3d.pair_certificates import broad_pairs, certified
from saft3d.validation import _intersection_kind, _FACE_INDEX, _CERT_MARGIN, _CERT_ROUNDOFF


def test_sweep_visits_every_aabb_pair_once():
    rng = np.random.default_rng(747)
    low = rng.uniform(-.5,.5,(173,3)); high = low+rng.uniform(.05,.45,(173,3))
    low[2]=high[1]+1e-11;high[2]=low[2]+.1  # contact inside tolerance band
    excluded=np.arange(len(low))%11==0;tol=1.e-10
    expected={(i,j) for i,j in combinations(range(len(low)),2)
              if not (excluded[i] or excluded[j]) and np.all(low[i]<=high[j]+tol)
              and np.all(low[j]<=high[i]+tol)}
    batches=list(broad_pairs(low,high,excluded,tol,chunk=17))
    actual=[tuple(pair) for batch in batches for pair in batch]
    assert len(actual)==len(set(actual)) and set(actual)==expected


def test_batched_certificates_never_clear_lp_penetration_or_unindexed_touch():
    rng=np.random.default_rng(832)
    base=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    vertices=[];cells=[];pairs=[];expected=[]
    for shared in (0,1,2,3,4):
        for i in range(5):
            shift=rng.uniform(-1,1,3);matrix=rng.normal(size=(3,3))
            a=base@matrix+shift; b=a.copy()
            if i==0: b[shared:]=a[shared:] # coincident coordinates, differing IDs
            elif i==1:b[shared:]=shift-.5*(a[shared:]-shift)
            else:b[shared:]+=rng.normal(size=(4-shared,3))*.7
            n=len(vertices)
            ids=np.arange(n,n+4);other=np.r_[ids[:shared],np.arange(n+4,n+8-shared)].astype(int)
            vertices.extend(a);vertices.extend(b[shared:]);pairs.append([len(cells),len(cells)+1]);cells.extend([ids,other])
            expected.append(_intersection_kind(a,b,ids,other)[0])
    t=np.array(cells);c=np.array(vertices)[t]
    cleared=certified(t,c,np.array(pairs),_FACE_INDEX,_CERT_MARGIN,_CERT_ROUNDOFF)
    assert cleared.any()
    assert all(not clear or kind in ('disjoint','conforming') for clear,kind in zip(cleared,expected))
    assert 'overlap' in expected and 'nonconforming contact' in expected


def test_batched_sat_matches_scalar_for_multiple_tiles_and_shared_face():
    rng=np.random.default_rng(190);a=rng.uniform(-.5,.5,(267,4,3))
    b=rng.uniform(-.5,.5,(70,4,3))+rng.uniform(-.6,.6,(70,1,3))
    a[::5]+=2
    b[0]=np.array([[0.,0.,0.],[1.,0,0],[0,1.,0],[0,0,1.]])
    a[1]=b[0]*[1.,1.,-1.]
    old_low,old_high=b.min(1),b.max(1);tol=1e-9
    expected=[]
    for t in a:
        cand=np.flatnonzero(np.all(np.minimum(t.max(0),old_high)-np.maximum(t.min(0),old_low)>tol,axis=1))
        expected.append(any(tet_overlap(t,b[j],tol) for j in cand))
    np.testing.assert_array_equal(overlap_any(a,b,tol),expected)
