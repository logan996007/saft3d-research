# Batched AFT and quality improvements

Benchmarks used an Intel Core i5-9300H and GTX 1660 Ti. The exact U-bracket seed from the 3,078-free-DOF experiment has 1,176 independent straight triangular prisms and 3,528 tetrahedra.

The previous implementation dispatched three local front iterations separately for each prism. The new implementation batches those independent fronts. Given a prism and its eight prescribed surface triangles, it evaluates all 15 choices of four vertices and updates a compact oriented boundary until it closes. The CPU backend works in NumPy batches. CUDA assigns one thread to each prism and returns the full connectivity to host memory. Incompatible diagonals and degenerate cells raise errors.

| 3,528-cell seed, one fresh process per backend | CPU (s) | CUDA with cached device code (s) |
|---|---:|---:|
| Original serial dispatch | 66.595 | 83.033 |
| Batched fronts | 0.03043 | 0.82753 |

Both first-call timings include decoding the input cap and all layer indices, backend setup and synchronized host results. A fresh CUDA process incurs setup cost. The warm timings below reuse the backend, include front construction, certificates, transfers and host results, and exclude the independent post-generation audit. Larger cases keep the same U cross-section and use more layers. They measure throughput on straight prisms with prescribed walls.

| Tetrahedra | CPU median (s) | CUDA median (s) | CPU / CUDA |
|---:|---:|---:|---:|
| 3,528 | 0.02293 | 0.01781 | 1.29× |
| 35,280 | 0.07280 | 0.01870 | 3.89× |
| 105,840 | 0.15651 | 0.02199 | 7.12× |
| 1,058,400 | 1.36902 | 0.04476 | 30.58× |

For a general closed surface, the original global AFT remains available. On the hollow cube, three-run median time went from 48.413 s to 4.207 s on CPU and from 10.978 s to 2.818 s on CUDA. All runs produced identical vertices and the same 201 tetrahedra. The sphere produced the same 571 cells on CUDA in 9.701 s versus 27.261 s in the previous single run. New CPU runs produced 589 cells in 15.370 s; compare like-for-like results when interpreting speedups.

The independent auditor batches spatial candidate pairs and verifies strict separation planes. Pairs without a verified certificate still go through the scalar checks and, where needed, the barycentric LPs. File export and independent audit are separate from mesh-generation time.

## Quality and FEM

The U-bracket quality optimizer imposes a relative compliance-loss budget and checks the final shape, domain, connectivity, and independent P1 solution. It can spend more time on optimization than on mesh generation. The resulting field is a separate, optional quality treatment of the goal-adapted mesh.

| U bracket | Before | After |
|---|---:|---:|
| Worst mean ratio | 0.00366 | 0.07624 |
| Fifth percentile mean ratio | 0.11223 | 0.16853 |
| Median mean ratio | 0.37986 | 0.37932 |
| Minimum dihedral angle | 0.114° | 2.364° |
| Mean loaded-face x displacement | 1.127782 mm | 1.125679 mm |

The final mesh retained 3,078 free P1 DOFs. Independent geometry validation reported zero overlaps or nonconforming contacts. Mean displacement errors against the two P2 references were 4.52% and 4.71%. The full optimizer took 113.95 s. The fast preset took 7.21 s and gave a minimum mean ratio of 0.04545. These costs are excluded from the seed-generation timings.

[Numerical data](optimization-data.json) · [Improved mesh](../output/speed-quality/quality-full/mesh.npz) · [Original scale study](performance.md)

## Reproduce

```sh
python -m fem_comparison.benchmark_batched_prism --layers 2 20 60 600 --repeats 5
python -m fem_comparison.benchmark_general --cases hollow-large sphere-stress --repeats 3 --timeout 120
python -m fem_comparison.improve_u_quality
python -m fem_comparison.improve_u_quality --fast --out output/speed-quality/quality-fast
```
