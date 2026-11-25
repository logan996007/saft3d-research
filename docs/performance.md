# Scale and CPU comparisons

Measured on one Intel Core i5-9300H (4 cores, 8 threads) with an NVIDIA GTX 1660 Ti (6 GiB). Times are wall-clock seconds for in-memory mesh generation. File export, independent verification and finite-element solutions are timed separately. The general AFT and the direct extrusion sweep perform different meshing operations. [Machine-readable measurements](benchmark-data.json)

## General advancing front

Each fresh process uses the same input and local AFT parameters with CPU or CUDA selected. Three trials per backend include initialization and transfers, using the existing CUDA disk cache. A ratio above one favors CUDA. The outputs of each paired case have identical connectivity, boundary keys and coordinates.

| Case | Tetrahedra | CPU median (s) | CUDA median (s) | CPU / CUDA |
|---|---:|---:|---:|---:|
| Cube, small | 12 | 1.043 | 1.659 | 0.63× |
| Cube, medium | 23 | 5.618 | 3.062 | 1.83× |
| Hollow cube | 201 | 48.413 | 10.978 | 4.41× |

For the hollow cube the candidate/accepted-cell overlap checks dropped from 38.19 s to 0.20 s. The CUDA path still spent 8.64 s checking legality against the active front on the CPU. The 128-face sphere stress case produced 571 cells in 27.26 s on CUDA, including 90 CPU-completed cells. This experiment measures a specific input and time budget.

For the actual 3,078-DOF U-bracket seed used by the accuracy study, each backend independently meshed 1,176 triangular prisms with local AFT. The 1,038 vertices and 3,528 tetrahedra matched exactly. The single measured CPU generation took **66.60 s** and CUDA **83.03 s** (CPU time / CUDA time **0.80×**). Each small prism generates only three tetrahedra and incurs repeated dispatch, setup and host-side checks. This measurement excludes the elasticity solves and later mesh optimization.

## Direct sweep on a straight U extrusion

The fast path starts from the same coarse 16 points and 28 triangles. It refines the cross-section and advances through prescribed layers with compatible prism diagonals. The table includes preparing the input, generating all cells, computing their volumes and mean ratios, and returning points, connectivity and the boundary to host memory. CPU refers to the NumPy implementation of this same sweep. Mesh arrays matched bit for bit at every size.

| Tetrahedra | CPU warm (s) | CUDA warm (s) | CPU / CUDA | Warm trials per backend |
|---:|---:|---:|---:|---:|
| 1,152 | 0.175 | 0.160 | 1.10× | 3 |
| 9,216 | 0.158 | 0.189 | 0.83× | 3 |
| 73,728 | 0.212 | 0.167 | 1.26× | 3 |
| 589,824 | 0.403 | 0.200 | 2.02× | 3 |
| 4,718,592 | 2.343 | 0.247 | 9.50× | 3 |
| 37,748,736 | 18.798 | 0.997 | 18.86× | 1 |

The last row returned 6,406,785 vertices and 1.28 GiB of mesh arrays. First calls from cold processes took 21.379 s on CPU and 2.407 s on CUDA. On the warm run, per-cell quality checks took 17.304 s on CPU and 0.193 s on CUDA. Connectivity itself took 1.171 s and 0.032 s respectively. An independent structure check visited all 12,582,912 prisms and compared full-array hashes outside the generation timer; it took another 6.47 s for the CPU result and 7.59 s for the CUDA result. The measured peak host working sets were 1.41 and 1.88 GiB respectively.

37.75 million is the largest verified run for this geometry and fast path. A 56.62-million-cell case exceeded the experiment's conservative device-memory budget and was not allocated. [73,728-cell VTK example](../examples/u_sweep_73728.vtk)

## Independent CPU TetGen

TetGen generated complete quality tetrahedral meshes on the same physical U domain, with a radius-edge ratio parameter of 1.8 and a maximum cell volume. It can insert points and refine boundary facets. This measures the cost of a different algorithm with different element distributions and quality objectives.

| Output cells | CPU generation (s) | Trials |
|---:|---:|---:|
| 1,123 | 0.01194 | 3 |
| 9,192 | 0.10070 | 3 |
| 77,049 | 1.044 | 3 |
| 608,449 | 10.557 | 3 |
| 1,039,245 | 20.749 | 3 |
| 4,878,630 | 99.098 | 1 |

The largest TetGen result had 823,321 nodes, median mean ratio 0.901 and used about 840 MiB peak host working set during generation. The time at around 4.8 million cells cannot be divided by the GPU sweep time to infer a hardware-only speedup. On the small frozen-boundary cube, native TetGen generated 12 tetrahedra in about 1.21 ms, substantially faster than either general research-AFT backend on that case.

## Reproduce

Run from the repository root. Use a quiet host for timing. Start with the small levels. The paired large sweep uses platform memory guards; the reported large-scale run was on Windows. An explicit level list controls larger allocations.

```sh
python -m fem_comparison.benchmark_general --cases cube-small cube-medium hollow-large --repeats 3
python -m fem_comparison.benchmark_general --stress --timeout 120
python -m fem_comparison.benchmark_general --prism-pair --mesh output/u-goal-adapt/p1-anisotropic/seed.npz --out output/prism-pair --timeout 300
python -m fem_comparison.benchmark_scale --levels 2 3 4
python -m fem_comparison.benchmark_tetgen --cases u_1k u_9k u_74k --repeats 3
```
