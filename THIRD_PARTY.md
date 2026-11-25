# Attribution and dependencies

This independent 3D implementation draws on the parallel candidate-generation and conflict-resolution approach proposed by Qingyi Zhou, Qiqi Wang and Zongfu Yu in *SAFT: Shotgun advancing front technique for massively parallel mesh generation on graphics processing unit*. The authors' original CUDA implementation generates 2D triangles. The paper and original source are available from the authors. [Paper](https://doi.org/10.1002/nme.7038) · [Original MIT-licensed project](https://github.com/zhouqingyi616/SAFT) · [Inspected revision](https://github.com/zhouqingyi616/SAFT/tree/3cd7ddf901a3aca80925716dba7d08fda7c802fd)

The MIT license at the repository root applies to the code and synthetic examples published here. Dependencies installed into your environment retain their own licenses.

- The residual-cavity worker and the independent CPU controls use the PyVista **tetgen** Python wrapper. The wrapper declares MIT for its own code. The **TetGen** engine by Hang Si has separate AGPL-3.0-or-later / commercial licensing terms. See both license files supplied in the `tetgen` distribution, including `src/tetgen-license`. [Wrapper](https://github.com/pyvista/tetgen) · [TetGen project and license](https://www.wias-berlin.de/software/tetgen/)
- NumPy and SciPy provide numerical array operations, optimization and sparse linear algebra. [NumPy](https://github.com/numpy/numpy) · [SciPy](https://github.com/scipy/scipy)
- CuPy compiles the CUDA C++ kernels with NVRTC. The CUDA Toolkit and NVIDIA drivers are installed separately. [CuPy](https://github.com/cupy/cupy) · [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
- The FEM experiments use scikit-fem for independent element assemblies and Matplotlib for figures. [scikit-fem](https://github.com/kinnala/scikit-fem) · [Matplotlib](https://github.com/matplotlib/matplotlib)
