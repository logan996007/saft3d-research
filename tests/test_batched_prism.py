from itertools import permutations
import numpy as np
import pytest

from saft3d.batched_prism import advance_batched_prisms
from saft3d.backend import CPUBackend,create_backend
from saft3d.thickness_front import _advance_with_local_aft
from fem_comparison.cpu_meshes import extract_boundary
from saft3d.validation import validate_mesh


@pytest.fixture(scope='module')
def gpu():
    cp=pytest.importorskip('cupy')
    try: count=cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        if exc.status in (3,35,100):pytest.skip('CUDA device unavailable')
        raise
    if not count:pytest.skip('CUDA device unavailable')
    return create_backend('cuda')


def extrude(xy,z=(0.,.21,1.)):
    xy=np.array(xy,float)
    points=np.vstack([np.column_stack((xy,np.full(len(xy),v))) for v in z])
    return points,np.arange(len(points)).reshape(-1,len(xy))


def cells_key(t): return set(map(tuple,np.sort(t,axis=1)))


def test_all_six_compatible_wall_orders_and_both_cyclic_fail(gpu):
    points,levels=extrude([[0,0],[1,0],[.1,1]],z=[0,1]);cap=np.array([[0,1,2]])
    expected_keys=set()
    for ranks in permutations(range(3)):
        flags=np.array([[ranks[i]<ranks[(i+1)%3] for i in range(3)]])
        cpu,stats=advance_batched_prisms(points,cap,levels,CPUBackend(),diagonals=flags)
        device,gstats=advance_batched_prisms(points,cap,levels,gpu,diagonals=flags)
        np.testing.assert_array_equal(cpu,device)
        assert stats['candidate_geometry_evaluations']==15 and gstats['kernel_calls']=={'batch_front':1}
        assert gstats['front_eligibility_tests']==45 and gstats['remaining_front_faces']==0
        expected_keys.add(tuple(sorted(cells_key(cpu))))
        assert validate_mesh(points,cpu,extract_boundary(cpu))['valid']
    assert len(expected_keys)==6
    for engine in (CPUBackend(),gpu):
        for flags in (np.ones((1,3),bool),np.zeros((1,3),bool)):
            with pytest.raises(ValueError,match='legal|closure'):advance_batched_prisms(points,cap,levels,engine,diagonals=flags)


def test_multiple_layers_match_general_aft_and_keep_global_ids(gpu):
    p,l=extrude([[0,0],[1,0],[1,1],[0,1]])
    c=np.array([[3,0,2],[0,1,2]])
    perm=np.random.default_rng(41).permutation(len(p));l=perm[l];p=p[np.argsort(perm)]
    general,report=_advance_with_local_aft(p,c,l,'cpu')
    assert report['cpu_closure_prisms']==0
    for engine in (CPUBackend(),gpu):
        t,r=advance_batched_prisms(p,c,l,engine)
        assert cells_key(general)==cells_key(t)
        assert validate_mesh(p,t,extract_boundary(general))['valid']
        assert r['prisms_meshed']==4


def test_reject_mismatch_bad_winding_and_wall_diagonals():
    p,l=extrude([[0,0],[1,0],[1,1],[0,1]])
    c=np.array([[0,1,2],[2,3,0]])
    with pytest.raises(ValueError,match='inverted'):advance_batched_prisms(p,c[:,::-1],l,'cpu')
    broken=p.copy();broken[l[1,1],0]+=.03
    with pytest.raises(ValueError,match='match'):advance_batched_prisms(broken,c,l,'cpu')
    diagonals=c<np.roll(c,-1,axis=1);diagonals[0,2]=not diagonals[0,2]
    with pytest.raises(ValueError,match='incompatible'):advance_batched_prisms(p,c,l,'cpu',diagonals=diagonals)


@pytest.mark.parametrize('xy',[
    [[0,0],[1,0],[0,1],[.1,.1],[.3,.1],[.1,.3]], # contained cap triangle
    [[0,0],[1,0],[0,1],[.4,0],[2,-1],[2,0]], # positive-length unindexed contact
    [[0,0],[1,0],[0,1],[.4,-1],[.8,0],[.4,1]], # crossing edges
])
def test_reject_nonembedded_cap(xy):
    p,l=extrude(xy)
    with pytest.raises(ValueError,match='overlap|nonconforming'):
        advance_batched_prisms(p,np.array([[0,1,2],[3,4,5]]),l,'cpu')


def test_reject_disconnected_cap_vertex_fans():
    p,l=extrude([[0,0],[1,0],[0,1],[-1,0],[0,-1]])
    with pytest.raises(ValueError,match='nonmanifold cap boundary vertex'):
        advance_batched_prisms(p,np.array([[0,1,2],[0,3,4]]),l,'cpu')
