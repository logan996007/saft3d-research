import numpy as np
import pytest

from saft3d.general_surfaces import make_general_surface
from saft3d.thickness_front import grow_thickness_front
from saft3d.validation import validate_mesh
from fem_comparison.thermal import solve


def test_shared_diagonals_on_nonuniform_layers_and_prescribed_surface():
    v,f,meta=make_general_surface('slender_s')
    p,t,b,report=grow_thickness_front(v,f,[0,.15,.6,1],backend='cpu')
    np.testing.assert_array_equal(p[:len(v)],v)
    assert report['closure_tets']==0 and t.shape==(90,4)
    check=validate_mesh(p,t,b,check_overlap=True)
    assert check['valid'],check['errors']
    assert check['positive_volume_sum']==pytest.approx(meta['analytic_volume'])
    # Every planar sidewall must have its own original rectangular area.
    sides=p[b[np.ptp(p[b,2],axis=1)>0]]
    areas=np.linalg.norm(np.cross(sides[:,1]-sides[:,0],sides[:,2]-sides[:,0]),axis=1)/2
    polygon=v[:12,:2]
    for a,c in zip(polygon,np.roll(polygon,-1,axis=0)):
        rel=sides[:,:,:2]-a; edge=c-a
        on_line=np.abs(rel[:,:,0]*edge[1]-rel[:,:,1]*edge[0])<1.e-10
        along=rel@edge/np.dot(edge,edge)
        on_wall=(on_line & (along>=-1.e-10) & (along<=1+1.e-10)).all(axis=1)
        assert areas[on_wall].sum()==pytest.approx(np.linalg.norm(edge)*meta['extrusion_height'])


def test_exact_dof_and_convergence_for_actual_prism():
    v,f,_=make_general_surface('slender_s')
    errors=[]
    for n in (5,9):
        p,t,b,_=grow_thickness_front(v,f,np.linspace(0.,1.,n+1),backend='cpu')
        r,_=solve(p,t,b)
        assert r['free_dofs']==(n-1)*12 and r['relative_heat_imbalance']<1.e-12
        errors.append((r['relative_l2_temperature_rise'],r['relative_l2_heat_flux']))
    assert errors[0][0]<.05 and errors[1][0]<errors[0][0]
    assert errors[1][1]<errors[0][1]


def test_invalid_caps_and_spacing():
    v,f,_=make_general_surface('slender_s')
    with pytest.raises(ValueError,match='increase strictly'):
        grow_thickness_front(v,f,[0,.4,.4,1],backend='cpu')
    v[12:,0]+=.01
    with pytest.raises(ValueError,match='match vertically'):
        grow_thickness_front(v,f,[0,.5,1],backend='cpu')


def test_default_thickness_route_actually_batches_and_checks_the_full_volume():
    v,f,_=make_general_surface('slender_s')
    p,t,b,report=grow_thickness_front(v,f,[0,.25,.5,1],backend='cpu')
    assert report['mode']=='batched' and report['prism_evidence']['remaining_front_faces']==0
    assert report['prism_evidence']['front_eligibility_tests']==45*report['prism_evidence']['prisms_meshed']
    assert validate_mesh(p,t,b)['valid']


def test_cold_auto_prism_route_does_not_initialize_a_device(monkeypatch):
    import saft3d.backend as backends
    monkeypatch.setattr(backends, '_cuda_ready', False)
    def fail(*args,**kwargs):
        raise AssertionError('small cold auto call initialized CUDA')
    monkeypatch.setattr(backends,'CUDABackend',fail)
    v,f,_=make_general_surface('slender_s')
    _,_,_,report=grow_thickness_front(v,f,[0,1],backend='auto')
    assert report['backend']=='cpu' and 'cold' in report['device']['selection']


def test_actual_cuda_local_aft_and_direct_sweep_have_same_prism_partition():
    cp=pytest.importorskip('cupy')
    try: count=cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError as exc:
        if exc.status in (3,35,100): pytest.skip('CUDA device unavailable')
        raise
    if count==0: pytest.skip('CUDA device unavailable')
    v,f,_=make_general_surface('slender_s')
    gp,gt,gb,run=grow_thickness_front(v,f,[0,.33,1],backend='cuda',mode='aft')
    cpu,ct,cb,_=grow_thickness_front(v,f,[0,.33,1],backend='cpu',mode='sweep')
    direct,dt,db,sweep=grow_thickness_front(v,f,[0,.33,1],backend='cuda',mode='sweep')
    np.testing.assert_array_equal(gp,cpu);np.testing.assert_array_equal(direct,cpu)
    np.testing.assert_array_equal(gb,cb);np.testing.assert_array_equal(db,cb)
    for cells in (gt,dt):
        assert sorted(map(tuple,np.sort(cells,axis=1)))==sorted(map(tuple,np.sort(ct,axis=1)))
    assert run['prism_evidence']['prisms_meshed']==20
    for name in ('propose_local_tets','evaluate_tets','aabb_conflicts','tet_overlaps'):
        assert run['prism_evidence']['kernel_calls'][name]>0
    assert sweep['advance_prisms_gpu_launches']==2
