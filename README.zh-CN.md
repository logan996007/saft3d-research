# SAFT3D Research

这是受二维 SAFT 启发的三维四面体网格研究实现。输入为闭合、有向的三角面，支持三角面 OBJ、STL 和 JSON。GPU 负责候选生成、评分和相交筛选，CPU 更新前沿。难以闭合的残腔可以交给 CPU TetGen，程序会逐面复核与已生成网格的接口。

## 运行

在仓库根目录和 Python 3.10–3.12 虚拟环境中运行。CUDA 路径使用 NVIDIA 显卡、CUDA 12.x 和 NVRTC。

```sh
python -m pip install -e ".[cuda,fem,test]"
python -m saft3d --shape hollow_cube --backend cuda --size .8 --max-new-vertices 32 --out output/hollow
python -m pytest -q
```

仅使用 CPU 时，安装 `.[fem,test]` 并指定 `--backend cpu`。可在浏览器打开输出的 `viewer.html` 剖视四面体，或用 ParaView 打开 `mesh.vtk`。本次运行的有效性、CPU 补全比例及质量数据见 `report.json`。

## 如何读实验结果

- 在 U 形件的 3,078 个一阶位移自由度下，误差驱动网格的加载端平均位移为 1.12778 mm，相对于上一轮二次参考解的误差为 4.34%。常规 CPU TetGen 网格误差为 13.83%。这些结果针对加载面平均位移，峰值应力需要单独做收敛验证。[精度实验](docs/accuracy.md)
- 中空立方体的通用 AFT 在 CPU 和 CUDA 上生成完全相同的 201 个四面体，三次计时中位数分别为 48.41 s 和 10.98 s。直挤出快速扫掠在同一台机器上实测至 37,748,736 个四面体，两种规模对应不同的几何假设。[规模与速度实验](docs/performance.md)
- 通用任意形状输入需要闭合、无自交、定向正确，当前输入检查上限为 2,048 个三角面。几何验证基于浮点容差；尺寸、最差单元质量和工业仿真精度需要针对实际问题检查。

原始二维 SAFT、相关论文和依赖许可证均见[第三方说明](THIRD_PARTY.md)。本仓库使用 [MIT 许可证](LICENSE)。
