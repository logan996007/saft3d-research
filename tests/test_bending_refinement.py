import numpy as np
from saft3d.bending_refinement import bending_size,refine_bending,fixed_end
from saft3d.general_surfaces import make_general_surface
from saft3d.geometry import tet_faces
from saft3d.validation import validate_mesh
from fem_comparison.cpu_meshes import audit_boundary


def test_surface_metric_shrinks_at_reentrant_corners_and_support():
    x=np.array([[1,1,0],[2,1,-.5],[.5,3,0],[2.8,2.8,0]])
    h=bending_size(x)
    assert np.all(h[:3]<.65*h[-1])
    v,f,_=make_general_surface('u_prism')
    new=refine_bending(v,f,f,52)
    assert np.count_nonzero(~fixed_end(new.vertices))==52
    assert audit_boundary(new.vertices,new.boundary,v,f)<1e-12
    np.testing.assert_array_equal(new.cells,new.boundary)


def test_full_edge_star_split_preserves_tetra_interfaces_and_trace():
    v=np.array([[0,0,0],[2,0,0],[0,.8,0],[0,.3,.4],[0,.3,-.4]],float)
    t=np.array([[0,1,2,3],[0,2,1,4]])
    f=np.array([face for cell in t for face in tet_faces(cell) if set(face)!={0,1,2}])
    new=refine_bending(v,t,f,17)
    check=validate_mesh(new.vertices,new.cells,new.boundary)
    assert check['valid'],check['errors']
    assert audit_boundary(new.vertices,new.boundary,v,f)<1e-12
    assert new.details['added_vertices']==12
