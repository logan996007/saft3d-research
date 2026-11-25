import struct
import numpy as np
import pytest
from saft3d.io import read_surface,write_vtk,write_msh
from saft3d.surfaces import tetrahedron
from saft3d.mesher import check_surface


def test_obj_negative_indices_and_attributes(tmp_path):
    path=tmp_path/'shape.obj'
    path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3/2/1 -2/3/1 -1/4/1\n')
    v,f=read_surface(path)
    np.testing.assert_array_equal(f,[[0,1,2]])
    assert v.shape==(3,3)


@pytest.mark.parametrize('face',['f 1 2 3 4','f 0 2 3','f 1 2 9'])
def test_obj_bad_faces(tmp_path,face):
    path=tmp_path/'bad.obj'; path.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n'+face)
    with pytest.raises(ValueError): read_surface(path)


@pytest.mark.parametrize('binary',[False,True])
def test_stl_exact_welding_and_orientation(tmp_path,binary):
    v,f=tetrahedron(); path=tmp_path/'tet.stl'
    if binary:
        data=b'solid misleading binary header'.ljust(80,b' ')+struct.pack('<I',len(f))
        for tri in v[f]: data+=struct.pack('<12fH',0,0,0,*tri.ravel(),0)
        path.write_bytes(data)
    else:
        text='solid tet\n'
        for tri in v[f]:
            text+='facet normal 0 0 0\nouter loop\n'
            text+=''.join('vertex '+' '.join(map(str,p))+'\n' for p in tri)
            text+='endloop\nendfacet\n'
        path.write_text(text+'endsolid tet\n')
    p,t=read_surface(path)
    assert p.shape==(4,3) and t.shape==(4,3)
    assert check_surface(p,t,1e-10)==pytest.approx(8/3)


def test_truncated_stl_rejected(tmp_path):
    p=tmp_path/'bad.stl';p.write_text('solid x\nfacet normal 0 0 1\nvertex 0 0 0\n')
    with pytest.raises(ValueError,match='truncated'):read_surface(p)


def test_tet_export_connectivity(tmp_path):
    v=np.eye(4,3); v[3]=[0,0,0]; tet=np.array([[3,0,1,2]])
    vtk=tmp_path/'m.vtk';msh=tmp_path/'m.msh'
    write_vtk(vtk,v,tet);write_msh(msh,v,tet,np.array([[0,1,2]]))
    assert '4 3 0 1 2' in vtk.read_text()
    assert 'CELL_TYPES 1\n10' in vtk.read_text()
    assert '2 4 2 2 1 4 1 2 3' in msh.read_text()
