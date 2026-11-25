# U-bracket fixed-budget experiment

Both the guiding P1 solution and the final P1 solution use **3,078 free displacement DOFs**. The U bracket occupies 7 m³ with 1 m thick arms. Clamp the left end, apply a uniform 1 MPa traction in the x direction to the right end, set Young's modulus to 200 GPa and Poisson's ratio to 0.30. Report the area-weighted mean x displacement of the loaded face.

| Mesh | Displacement (mm) | Relative error |
|---|---:|---:|
| Geometry-based SAFT refinement | 1.014319 | 13.97% |
| Independent conventional CPU TetGen mesh | 1.015887 | 13.83% |
| CPU TetGen followed by the same three cycles of FE-driven flips and relocation | 1.101831 | 6.54% |
| CPU TetGen on every final adapted vertex and prescribed boundary triangle | 1.051767 | 10.79% |
| P1-error-driven cross-section, local connectivity changes and relocation | **1.127782** | **4.34%** |

Recover variation of the P1 displacement gradient using weighted least squares over 32 nearby elements, project this metric onto the cross-section, and evaluate six layouts under the same free-node budget. The selected layout uses three thickness planes. In a separate reconstruction, the full GPU local AFT independently generated the same 3,528 seed tetrahedra across all 1,176 triangular prisms. The optimization makes local tetrahedron swaps, then moves nodes tangentially on each boundary patch to reduce compliance discretization error. The reference value is used only after generation.

The primary numerical P2 reference was 1.178976 mm. A further P2 solve with 126,159 free DOFs gave 1.181283 mm, for a sensitivity-check error of 4.53%. These enriched solves are independent accuracy references. The final 3,527 tetrahedra passed volume, boundary, overlap and orientation checks. A second P1 assembly reproduced the displacement. The minimum mean ratio was 0.00366; local peak stress has a separate convergence requirement. [Independent audit](../output/u-goal-adapt/p1-polished/audit.json) · [Same-node CPU audit](../output/u-goal-adapt/p1-cpu-same/audit.json)

Run these commands from the repository root with the CUDA and FEM extras installed. To repeat only the CPU optimization and P1 audit, start at the third command using the included synthetic seed.

```sh
python -m fem_comparison.anisotropic_u --quick --pilot p1
python -m fem_comparison.verify_aft_seed --mesh output/u-goal-adapt/p1-anisotropic/seed.npz --out output/reproduced-aft.json
python -m fem_comparison.goal_polish output/u-goal-adapt/p1-anisotropic/seed.npz output/reproduced --cycles 3 --steps 75 --flips 6
python -m fem_comparison.audit_goal_u output/reproduced/mesh.npz output/reproduced
python -m fem_comparison.goal_cpu_control --mesh output/reproduced/mesh.npz --out output/reproduced-cpu
```

This is an accuracy experiment. Its cost includes repeated CPU finite-element solves; the generation throughput benchmark measures a different workload.
