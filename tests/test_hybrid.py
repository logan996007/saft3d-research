import numpy as np
import pytest
from saft3d.general_surfaces import make_general_surface
from saft3d.surfaces import torus
from saft3d.general_mesher import AdvancingFrontConfig,grow_general_mesh
from saft3d.validation import validate_mesh


@pytest.mark.parametrize('shape',['u_prism','hollow_cube','slender_s','torus'])
def test_aft_cells_retained_with_exact_cpu_cavity_interface(shape):
    if shape=='torus':
        v,f=torus(6,4)
    else: v,f,_=make_general_surface(shape)
    # A deliberately short AFT budget exercises the stitching boundary.
    cfg=AdvancingFrontConfig(backend='cpu',max_iterations=1,max_new_vertices=0,
                             size=.7,completion='tetgen')
    points,tets,report=grow_general_mesh(v,f,cfg)
    assert report['growth_tets']>0
    assert report['closure_tets']>0
    assert report['remaining_front_faces']==0
    assert report['cpu_completion']['boundary_exact']
    assert report['cpu_completion']['global_prefix_unchanged']
    assert 0<report['aft_volume_fraction']<1
    np.testing.assert_array_equal(points[:len(v)],v)
    result=validate_mesh(points,tets,f)
    assert result['valid'],result['errors']


def test_hybrid_translation_scale_invariance():
    v,f,_=make_general_surface('u_prism')
    cfg=AdvancingFrontConfig(backend='cpu',max_iterations=1,max_new_vertices=0,size=1,completion='tetgen')
    p,t,r=grow_general_mesh(v,f,cfg)
    cfg.size=1e3
    p2,t2,r2=grow_general_mesh(v*1e3+[51,-25,93],f,cfg)
    assert r2['cpu_completion']['boundary_exact']
    result=validate_mesh(p2,t2,f)
    assert result['valid'],result['errors']
    assert r2['aft_volume_fraction']==pytest.approx(r['aft_volume_fraction'],abs=1e-10)
