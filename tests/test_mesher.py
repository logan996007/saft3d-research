import numpy as np
import pytest
from saft3d.surfaces import cube,sphere,torus
from saft3d.mesher import grow_mesh,MesherConfig,UnsupportedDomain
from saft3d.validation import validate_mesh


@pytest.mark.parametrize("shape",[cube,lambda:sphere(0),lambda:sphere(1,True)])
def test_closed_star_domains(shape):
    v,f=shape()
    p,t,run=grow_mesh(v,f,MesherConfig(growth_rounds=1,size=.65))
    report=validate_mesh(p,t,f)
    assert report["valid"], report["errors"]
    assert run["remaining_front_faces"]==0
    assert run["growth_tets"]>0 and run["closure_tets"]>0


def test_torus_rejected():
    v,f=torus()
    with pytest.raises(UnsupportedDomain,match="visibility kernel"):
        grow_mesh(v,f)


def test_open_boundary_rejected():
    v,f=cube()
    with pytest.raises(ValueError,match="closed"):
        grow_mesh(v,f[:-1])


def test_dented_fixture_is_nonconvex():
    v,f=sphere(1,True)
    tri=v[f]
    normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    distances=np.einsum('fvi,fi->fv',v[None,:,:]-tri[:,None,0,:],normals)
    assert distances.max()>1e-3


def test_bad_indices_and_tolerance_rejected():
    v,f=cube()
    with pytest.raises(ValueError,match="integer"):
        grow_mesh(v,f.astype(float)+.1)
    with pytest.raises(ValueError,match="tolerance"):
        grow_mesh(v,f,MesherConfig(tolerance=float('nan')))


def test_translation_and_scale_invariance():
    v,f=cube()
    p,t,_=grow_mesh(v,f,MesherConfig(growth_rounds=0))
    p2,t2,_=grow_mesh(v*100+np.array([40,5,-9]),f,MesherConfig(growth_rounds=0,size=60))
    np.testing.assert_array_equal(t,t2)
    np.testing.assert_allclose(p2,p*100+[40,5,-9],atol=1e-12)
