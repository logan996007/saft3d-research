# SAFT3D Research

GPU-assisted generation of tetrahedral volume meshes from closed triangular surfaces, with a CPU reference backend, independent mesh validation and reproducible finite-element experiments. [中文说明](README.zh-CN.md)

The general mesher grows tetrahedra inward from an oriented triangle front. CUDA kernels propose vertices, score tetrahedra and check conflicts. CPU code checks geometric legality, selects independent candidates and updates the front. If advancement stalls, the default mode passes the remaining cavity to a CPU TetGen worker and verifies that every interface triangle is preserved. Reports record the contributions of both methods. [Methods and attribution](THIRD_PARTY.md)

## Get started

Clone the repository and run the following commands from its root with Python 3.10–3.12. A virtual environment is recommended.

```sh
python -m pip install -e ".[fem,test]"
python -m saft3d --shape cube --backend cpu --max-new-vertices 1 --out output/demo
python -m pytest -q
```

For an NVIDIA GPU with a CUDA 12.x Toolkit and NVRTC, install the CUDA extra and select the device explicitly.

```sh
python -m pip install -e ".[cuda,fem,test]"
python -m saft3d --shape hollow_cube --backend cuda --size .8 --max-new-vertices 32 --out output/hollow
python -m saft3d --input examples/torus_coarse.json --backend cuda --size .5 --max-new-vertices 32 --out output/torus
```

Open `mesh.vtk` in ParaView, `mesh.msh` in Gmsh, or `viewer.html` in a browser. Look at `report.json` for mesh and device diagnostics. The input may be a triangle OBJ, STL, or a JSON object with `vertices` and zero-based, outward-oriented `faces`. Cavity shells face into the void. A failed generation or validation returns a nonzero exit code. For experiments with pure advancing front, use `--completion none`; a stalled run records its partial mesh.

## Geometry and validation

- **General domains.** Local front vertices can merge across arms and shells. The front may become a singular oriented chain during valid transitions. Examples include a U bracket, a hollow cube and a torus.
- **Extrusions.** Matching planar caps support thickness-directed layers. The `thickness` CLI method partitions each layer into triangular prisms and runs local AFT. A separate direct staircase sweep in the scale benchmark uses predefined connectivity for straight extrusions.
- **Checks.** Validation checks signed volumes, orientation, boundary matching and volume conservation. Broad-phase filtering and separating-plane certificates remove simple pairs; independent barycentric linear programs check remaining pairs for penetration and nonconforming contact. Predicates use floating-point tolerances.
- **Limits.** The general input checker supports up to 2,048 boundary triangles. The solver uses a local size heuristic with fixed boundary facets. Slivers, stalled fronts and expensive host-side legality checks remain possible. Inspect the outputs and check convergence for your physical observable.

## Research results

On a bending U bracket with 3,078 free P1 displacement DOFs, error-driven redistribution of vertices and local connectivity optimization reduced the loaded-face displacement error from 13.97% to 4.34% against a refined P2 reference. The corresponding conventional CPU TetGen mesh gave 13.83%. This comparison uses the same material, boundary conditions and P1 solver. [Accuracy experiment and reproduction](docs/accuracy.md) · [Audited P1 result](output/u-goal-adapt/p1-polished/audit.json)

On an i5-9300H and GTX 1660 Ti, general AFT generated an identical 201-cell hollow mesh in 48.41 s with the project's CPU backend and 10.98 s with CUDA. A regular U extrusion with 37,748,736 tetrahedra took 18.80 s on NumPy and 1.00 s on CUDA in the direct sweep benchmark, including cell checks and return to host memory. These experiments measure different algorithms and workloads. [Scale, timing scopes and CPU TetGen comparisons](docs/performance.md)

The published fixtures are synthetic. Use a repository checkout and editable install to run the experiments that read these fixtures. The Python wheel contains the library and CUDA kernels. The generator, validation and tests are in `saft3d/` and `tests/`. Benchmark drivers and finite-element solvers are in `fem_comparison/`. [Source and fixture provenance](provenance.json)

MIT license for this repository. TetGen carries separate licensing terms. [License](LICENSE) · [Third-party notices](THIRD_PARTY.md)
