"""Conservative front certificates, cache lifetime and local topology changes."""
from itertools import permutations
import numpy as np
import pytest
from saft3d.front_geometry import FrontGeometry,FrontContactCache
from saft3d.general_mesher import _front_delta,_updated_front
from saft3d.general_surfaces import make_general_surface
from saft3d.geometry import tet_faces
from saft3d.surfaces import cube


class ScalarFront(FrontGeometry):
    """Force every AABB pair through the retained pre-optimization narrow phase."""
    def _batch_certificates(self,tet,ids,normals,anchors,nearby):
        return np.zeros(len(nearby),dtype=bool)


@pytest.mark.parametrize('hollow',[False,True])
def test_batched_plane_certificates_match_scalar_front(hollow):
    p,f=make_general_surface('hollow_cube')[:2] if hollow else cube()
    p=(p-p.min(0))/np.ptp(p,axis=0).max()
    new,old=FrontGeometry(p,f),ScalarFront(p,f)
    rng=np.random.default_rng(9184)
    for i in range(84):
        j=i%len(f); tri=f[j, [0,2,1]]; a,b,c=p[tri]
        apex=(a+b+c)/3 - new.normals[j]*rng.choice([1.e-12,.06,.25,1.02])
        idx=-1
        if i%3==0:idx=int(rng.choice(np.setdiff1d(np.arange(len(p)),tri)));apex=p[idx]
        if i%3==1:apex=apex+rng.normal(0,.08,size=3)
        coords=np.vstack((p[tri],apex)); ids=[*tri,idx]
        assert new.valid_candidate(coords,ids)==old.valid_candidate(coords,ids)
    assert new.stats['certified_pairs']>0
    assert new.stats['lp_calls']<=old.stats['lp_calls']


def test_cache_checks_new_front_facets_and_new_vertices():
    v,f=cube();v=(v+1.)/2;face=f[0]
    cache=FrontContactCache()
    def snapshot(p,faces):
        g=FrontGeometry(p,faces);g.contact_cache=cache;g.face_tokens=cache.face_tokens(faces)
        return g
    start=snapshot(v,f);tri=face[[0,2,1]]
    apex=v[face].mean(0)-start.normals[0]*.45
    tet=np.vstack((v[tri],apex));ids=[*tri,-1]
    assert start.valid_candidate(tet,ids)
    same=snapshot(v,f)
    assert same.valid_candidate(tet,ids) and same.stats['reused_pairs']>0
    # New inward-facing shell lies inside the candidate but away from its
    # centroid. Cached checks of original cube faces cannot admit this tet.
    origin=.75*apex+.25*tet[0]; tiny=origin+(v-.5)*.003
    new_faces=np.vstack((f,f[:,::-1]+len(v)))
    changed=snapshot(np.vstack((v,tiny)),new_faces)
    assert changed.contains(tet.mean(0))
    assert not changed.valid_candidate(tet,ids)
    assert snapshot(v,f).valid_candidate(tet,ids)
    dup=snapshot(np.vstack((v,apex)),f)
    assert not dup.valid_candidate(tet,ids)
    assert dup.stats['last_rejection']['reason']=='new_vertex_duplicates_existing_vertex'


def test_tet_delta_equals_checked_global_chain_update():
    f={tuple(sorted(t)):t for t in tet_faces((0,1,2,3))}
    for indices in permutations(range(5),4):
        old=_updated_front(f,indices);delta=_front_delta(f,indices)
        if old is None: assert delta is None;continue
        result=f.copy()
        for key,face in delta:
            if face is None:del result[key]
            else:result[key]=face
        assert list(result.items())==list(old.items())


def test_block_query_covers_long_faces_and_matches_exhaustive_order():
    v,f=cube()
    # One long crossing shell and many short, separated ones. Centroid-only
    # distance pruning could miss a long face close to its endpoint.
    parts=[v*[80.,1.,1.]]
    parts.extend(v*.25+[i,10,0] for i in range(17))
    g=FrontGeometry(np.vstack(parts),np.vstack([f+i*len(v) for i in range(len(parts))]))
    assert g.block_order is not None
    rng=np.random.default_rng(313)
    for origin in np.vstack(([79.,-.9,-1.],rng.uniform([-90,-5,-5],[90,15,5],(64,3)))):
        high=origin+[.45,2.6,5.]
        expected=np.flatnonzero(np.all((high+g.tol>=g.lows)&(g.highs+g.tol>=origin),axis=1))
        np.testing.assert_array_equal(expected,g._nearby_faces(origin,high))
