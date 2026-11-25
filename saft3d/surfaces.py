"""Small, deterministic triangulated domains. No volume mesher is used here."""
from __future__ import annotations
import numpy as np


def _orient(vertices, faces, interior=(0.0, 0.0, 0.0)):
    faces = np.asarray(faces, dtype=np.int64).copy()
    p = np.asarray(vertices, dtype=float)
    for i, f in enumerate(faces):
        a, b, c = p[f]
        if np.dot(np.cross(b-a, c-a), np.asarray(interior)-a) > 0:
            faces[i] = f[[0, 2, 1]]
    return p, faces


def cube():
    v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                  [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], dtype=float)
    quads = [(0,1,2,3),(4,7,6,5),(0,4,5,1),(1,5,6,2),(2,6,7,3),(3,7,4,0)]
    f = [(a,b,c) for a,b,c,d in quads] + [(a,c,d) for a,b,c,d in quads]
    return _orient(v, f)


def tetrahedron():
    return _orient(np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]],float),
                   [(0,1,2),(0,3,1),(0,2,3),(1,3,2)])


def sphere(subdivisions=1, dented=False):
    if not 0 <= subdivisions <= 3:
        raise ValueError("sphere subdivisions must be in [0, 3]")
    vertices = [[1.,0,0],[-1.,0,0],[0,1.,0],[0,-1.,0],[0,0,1.],[0,0,-1.]]
    faces = [(0,2,4),(2,1,4),(1,3,4),(3,0,4),
             (2,0,5),(1,2,5),(3,1,5),(0,3,5)]
    for _ in range(subdivisions):
        mids = {}
        def mid(a,b):
            key = tuple(sorted((a,b)))
            if key not in mids:
                p = np.asarray(vertices[a]) + vertices[b]
                mids[key] = len(vertices)
                vertices.append((p / np.linalg.norm(p)).tolist())
            return mids[key]
        refined = []
        for a,b,c in faces:
            ab,bc,ca = mid(a,b),mid(b,c),mid(c,a)
            refined += [(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]
        faces = refined
    v = np.asarray(vertices)
    if dented:
        # Radial deformation retains the origin in the visibility kernel.
        radius = 1.0 - 0.68 * np.exp(-((v[:,0]-1)**2 + v[:,1]**2 + v[:,2]**2)/0.45)
        v *= radius[:,None]
    return _orient(v, faces)


def torus(n_major=12, n_minor=8):
    vertices, faces = [], []
    for i in range(n_major):
        u = 2*np.pi*i/n_major
        for j in range(n_minor):
            t = 2*np.pi*j/n_minor
            vertices.append([(1+0.35*np.cos(t))*np.cos(u),
                             (1+0.35*np.cos(t))*np.sin(u),0.35*np.sin(t)])
    for i in range(n_major):
        for j in range(n_minor):
            a=i*n_minor+j; b=((i+1)%n_major)*n_minor+j
            c=((i+1)%n_major)*n_minor+(j+1)%n_minor; d=i*n_minor+(j+1)%n_minor
            faces += [(a,b,c),(a,c,d)]
    return np.asarray(vertices), np.asarray(faces,dtype=np.int64)


def make_surface(name, subdivisions=1):
    if name == "cube": return cube()
    if name == "tetrahedron": return tetrahedron()
    if name in ("sphere", "dented"):
        return sphere(subdivisions, name == "dented")
    if name == "torus": return torus()
    raise ValueError(f"unknown surface: {name}")
