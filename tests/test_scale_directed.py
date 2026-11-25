import numpy as np
import pytest

from fem_comparison.benchmark_scale import verify_sweep
from saft3d.general_surfaces import make_general_surface
from saft3d.thickness_front import grow_thickness_front
from saft3d.validation import validate_mesh


def test_uniform_cap_refinement_conforms_and_keeps_geometry():
    v,f,meta=make_general_surface('u_prism')
    p,t,b,r=grow_thickness_front(v,f,[0,.5,1],backend='cpu',mode='sweep',cap_refinements=1)
    assert len(t)==144
    np.testing.assert_array_equal(p[:len(v)],v)
    assert r['total_volume']==pytest.approx(meta['analytic_volume'])
    assert r['min_volume']>0
    assert verify_sweep(p,t,b,1,2)['all_prisms_checked']==48
    assert validate_mesh(p,t,b)['valid']
    damaged=t.copy();damaged[0]=damaged[1]
    with pytest.raises(RuntimeError,match='incorrect staircase'):
        verify_sweep(p,damaged,b,1,2)
