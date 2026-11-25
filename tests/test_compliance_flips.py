"""Regression checks for the variational swap criterion."""
import numpy as np
import pytest
from saft3d.compliance_flips import _energy, _rings


def test_affine_elastic_field_has_no_energy_preference_between_valid_patches():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[.25,.25,1.],[.25,.25,-.5]])
    original=np.array([[0,1,2,3],[1,0,2,4]])
    candidates=list(_rings(original))
    assert len(candidates)==1
    _, replacement=candidates[0]
    u=p@np.array([[.001,.0003,0.],[0.,-.003,.0002],[0.,0.,.002]])
    e0,v0,_=_energy(p,original,u)
    e1,v1,_=_energy(p,np.asarray(replacement),u)
    assert v0.sum()==pytest.approx(v1.sum())
    assert e0.sum()==pytest.approx(e1.sum(),rel=2.e-14)


def test_degenerate_replacement_cannot_improve_energy():
    p=np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[1.,1.,0.]])
    e,v,_=_energy(p,np.array([[0,1,2,3]]),np.zeros_like(p))
    assert v[0]==0 and np.isposinf(e[0])
