# GPU Thermoworkflows

本项目将 Ni Tutton 研究中使用的执行与后处理步骤整理为独立工具。
提供 CP2K、GPU4PySCF 和离线合成模型三个后端，统一处理配置、固定结构力点、
暂停、续算、来源校验、数值 Hessian 与振动热修正。

CUDA 计算能力由 CP2K / GPU4PySCF 上游提供。本项目的贡献是工作流与实际部署经验。
当前版本为 v0.1.0。仓库：https://github.com/zhangxinxin528-afk/gpu-thermoworkflows 。

适用于分子或晶体的固定结构力点批量计算、长队列暂停续算、CPU/GPU 一致性比较、
数值 Hessian、振动频率与指定构型的热修正。更换结构与配置即可复用执行步骤；
具体化学体系的物种、电子方法和溶液模型需要由使用者选择。

## 快速开始

```sh
git clone https://github.com/zhangxinxin528-afk/gpu-thermoworkflows.git
cd gpu-thermoworkflows
python -m venv .venv
```

激活环境：Linux / WSL 执行 `source .venv/bin/activate`；Windows PowerShell
执行 `.\.venv\Scripts\Activate.ps1`。然后安装并运行：

```sh
python -m pip install .
gtw doctor
gtw run examples/water/mock.json
gtw analyze examples/water/mock.json
```

默认小例子只需要 Python 和 NumPy，计算 19 个合成力点，检查安装、执行、文件保存
和分析是否正常。合成模型结果会明确标记 `synthetic: true`。

[GitHub Release](https://github.com/zhangxinxin528-afk/gpu-thermoworkflows/releases/tag/v0.1.0)
提供源码压缩包与 wheel 安装包；示例和编译说明位于源码包中。

真实计算例子：

- `examples/water/cp2k.json`：一个固定水分子 CP2K 能量/力点。
- `examples/water/gpu4pyscf.json`：一个固定水分子 GPU4PySCF / SMD 能量/力点。

Linux / WSL 内安装真实引擎后，分别执行 `gtw run 配置文件`。
CP2K 例子需要把 `CP2K_DATA_DIR` 设置为该安装的基组和赝势数据目录。
这些是小型安装检查，不采用 Ni Tutton 的高截断生产预算。

## 修改为自己的体系

复制一个配置文件，再修改 XYZ 结构与 JSON。路径相对配置文件解析。
结构、方法、位移步长、晶胞、线程、并发、临时目录、温度等均有独立配置字段。
CP2K 的元素基组与赝势写在 `method.kinds` 中，复杂输入可提供自己的模板。

常用命令：

```sh
gtw prepare 配置.json
gtw run 配置.json
gtw status 配置.json
gtw pause 配置.json
gtw resume 配置.json
gtw analyze 配置.json --temperature 298.15
```

`pause` 在第二个终端调用：停止提交新点，等正在运行的点保存后进入 `PAUSED`。
确认暂停完成后再关闭计算终端或电脑。中断点的原尝试和日志保留，恢复时重新计算该点；
已完成点通过指纹、坐标及文件哈希检查后复用。
资源配置可以调整，物理输入或引擎版本变化时使用新的输出目录。

本工具不调用几何优化。它输出指定构型的热修正；完整溶解度仍需要另外定义热力学循环、
水合物计量、溶液标准态和活度模型。可以复用执行工具，不能直接替换成另一种盐的溶解度。

## 已测试资料与发布材料

- [历史实测与新工具测试](docs/benchmarks.md)
- [配置接口](docs/configuration.md)
- [CUDA 编译步骤](docs/cuda-build.md)
- [上游许可证和引用](THIRD_PARTY_NOTICES.md)
- [代码引用元数据](CITATION.cff)
- [英文完整说明](README.md)

原工作流代码采用 MIT；CP2K 源码及二进制仍遵循其 GPL。
本包只发布自己的脚本、示例与记录，没有捆绑 CP2K / CUDA 二进制。
