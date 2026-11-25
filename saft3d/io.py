"""Portable JSON input, legacy VTK and Gmsh 2.2 tetrahedral mesh output."""
from pathlib import Path
import json
import struct
import re
import numpy as np


def read_surface(path):
    path=Path(path)
    if path.suffix.lower()=='.obj': return _read_obj(path)
    if path.suffix.lower()=='.stl': return _read_stl(path)
    if path.suffix.lower()!='.json': raise ValueError('supported surface formats: .json, triangular .obj, .stl')
    data=json.loads(path.read_text(encoding="utf-8-sig"))
    vertices=np.asarray(data["vertices"],dtype=float)
    raw=np.asarray(data["faces"])
    if raw.dtype.kind not in "iu": raise ValueError("surface faces must be integer vertex indices")
    return vertices,raw.astype(np.int64)


def _read_obj(path):
    vertices=[]; faces=[]
    for line_no,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
        words=line.split('#',1)[0].split()
        if not words: continue
        if words[0]=='v':
            if len(words)!=4: raise ValueError(f'OBJ line {line_no}: require three vertex coordinates')
            p=list(map(float,words[1:]))
            if not np.isfinite(p).all(): raise ValueError('OBJ contains nonfinite coordinates')
            vertices.append(p)
        elif words[0]=='f':
            if len(words)!=4: raise ValueError(f'OBJ line {line_no}: only triangular faces are supported')
            ids=[]
            for word in words[1:]:
                i=int(word.split('/')[0])
                if i==0: raise ValueError('OBJ vertex indices are one-based; zero is invalid')
                i=i-1 if i>0 else len(vertices)+i
                if not 0<=i<len(vertices): raise ValueError(f'OBJ line {line_no}: vertex index out of range')
                ids.append(i)
            faces.append(ids)
    if not vertices or not faces: raise ValueError('OBJ has no triangular surface')
    return np.asarray(vertices,float),np.asarray(faces,np.int64)


def _read_stl(path):
    data=path.read_bytes()
    binary=False
    if len(data)>=84:
        count=struct.unpack_from('<I',data,80)[0]
        binary=(len(data)==84+50*count)
    triangles=[]
    if binary:
        for i in range(count):
            record=struct.unpack_from('<12fH',data,84+50*i)
            triangles.append(np.asarray(record[3:12]).reshape(3,3))
    else:
        try: text=data.decode('ascii')
        except UnicodeDecodeError as exc: raise ValueError('invalid or truncated binary STL') from exc
        groups=re.findall(r'\bfacet\s+normal\b.*?\bendfacet\b',text,flags=re.S|re.I)
        for group in groups:
            rows=re.findall(r'\bvertex\s+([^\r\n]+)',group,re.I)
            if len(rows)!=3: raise ValueError('ASCII STL facet must have three vertices')
            triangle=[list(map(float,row.split())) for row in rows]
            if any(len(row)!=3 for row in triangle): raise ValueError('STL vertex must have three coordinates')
            triangles.append(triangle)
        if len(re.findall(r'\bfacet\s+normal\b',text,re.I))!=len(groups):
            raise ValueError('truncated ASCII STL facet')
    if not triangles: raise ValueError('STL has no triangles')
    points=[]; faces=[]; ids={}
    for triangle in triangles:
        face=[]
        for point in triangle:
            key=tuple(map(float,point))
            if not np.isfinite(key).all(): raise ValueError('STL contains nonfinite coordinates')
            if key not in ids:
                ids[key]=len(points); points.append(key)
            face.append(ids[key])
        faces.append(face)
    # STL repeats vertices per facet; exact welding preserves its geometry.
    # No automatic near-point merge, normal flip, hole repair or surface filling.
    return np.asarray(points,float),np.asarray(faces,np.int64)


def write_vtk(path,vertices,tets,title='SAFT-inspired tetrahedral mesh'):
    with Path(path).open("w",encoding="ascii",newline="\n") as out:
        out.write(f"# vtk DataFile Version 3.0\n{title}\nASCII\nDATASET UNSTRUCTURED_GRID\n")
        out.write(f"POINTS {len(vertices)} double\n")
        for p in vertices: out.write(" ".join(f"{x:.17g}" for x in p)+"\n")
        out.write(f"CELLS {len(tets)} {5*len(tets)}\n")
        for tet in tets: out.write("4 "+" ".join(map(str,tet))+"\n")
        out.write(f"CELL_TYPES {len(tets)}\n"+"10\n"*len(tets))


def write_msh(path,vertices,tets,boundary):
    with Path(path).open("w",encoding="ascii",newline="\n") as out:
        out.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n")
        out.write(f"{len(vertices)}\n")
        for i,p in enumerate(vertices,1): out.write(f"{i} "+" ".join(f"{x:.17g}" for x in p)+"\n")
        out.write(f"$EndNodes\n$Elements\n{len(tets)+len(boundary)}\n")
        for i,tri in enumerate(boundary,1): out.write(f"{i} 2 2 1 1 "+" ".join(str(int(x)+1) for x in tri)+"\n")
        for i,tet in enumerate(tets,len(boundary)+1): out.write(f"{i} 4 2 2 1 "+" ".join(str(int(x)+1) for x in tet)+"\n")
        out.write("$EndElements\n")


def write_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf-8")
