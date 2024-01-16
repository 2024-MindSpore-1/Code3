# QuSmoke

基于 Mindspore 开发的量子模拟器。

项目来源：[2023昇腾AI创新大赛 第一批赛题 赛题八：使用MindSpore机器学习框架实现量子模拟](https://www.hiascend.com/zh/developer/contests/details/7e51ee21cd604773b5b91b974cef30be)

## 题目介绍

在NISQ阶段，变分量子算法是最有可能具有使用价值的算法。在变分量子算法里，我们需要学习量子线路中的参数，并使得线路的测量结果跟目标解决，因此我们需要利用梯度下降的算法来完成学习任务。而mindspore架构具有自动微分的能力，因此我们想要利用mindspore中的tensor作为基本数据类型、利用mindspore中的各种算子来完成量子模拟任务。此时在mindspore框架下，整个量子算法流程都是可微分的，能够达成量子机器学习的任务。

- mindspore机器学习框架使用便捷，具有高度的自定义性，且多平台适用。
- 考虑利用mindspore来完成量子模拟器，并自动利用mindspore架构完成量子线路的梯度计算。

## 环境要求

实现的量子模拟器 QuSmoke 支持以下硬件平台：

- CPU
- GPU
- Ascend

实现的量子模拟器 QuSmoke 开发时软件配置：

- Python >= 3.7
- Mindspore == 2.0

## 实现功能

- 实现常见量子逻辑门，包括
  - 单量子比特门：`H`，`X`，`Y`，`Z`，`RX`，`RY`，`RZ`，`T`，`SWAP`，`ISWAP`，`U1`，`U2`，`U3`
  - 多量子比特门：所有单量子比特门受控形式，支持单比特或多比特控制量子门。例如可通过单比特或两比特控制 `X` 门实现 `CNOT` 或 `Toffoli` 门。
- 量子线路幅值计算、实现哈密顿量测量等功能。
- 利用 Mindspore 的自动微分实现变分量子算法，能直接适用 Mindspore 的 `nn.Adam` 等优化器进行梯度更新。

开发的QuSmoke 模拟器具有如下特点：

- 基于 Mindspore 开发，量子线路采用深度学习的架构，将每个量子逻辑门类比经典神经网络层，用户可以像使用经典神经网络一样进行量子线路设计。
- 支持 Mindspore 静态图。
- 接口尽量与 mindquantum 保持一致，降低从 mindquantum 到 qusmoke 的学习成本。

## 文件结构

其文件主要包括 `qusmoke/` 文件夹和 `demo/` 文件夹，`qusmoke/` 下主要包括模拟器开发功能，`demo/` 为提供的一些案例教程。

```text
.
│  README.md
│
├─qusmoke/
│      circuit.py               # 线路模块
│      define.py                # 全局参数定义
│      expect.py                # 哈密顿量/期望值模块
│      gates.py                 # 量子逻辑门模块
│      operations.py            # 针对复数进行的操作模块
│      utils.py                 # 辅助功能函数
│
└─examples/
        demo_basic.ipynb                          # 基本操作
        demo_classification_of_iris_by_qnn.ipynb  # 鸢尾花二分类
        demo_qaoa_for_maxcut.ipynb                # QAOA 解决最大割问题
        demo_qnn_for_nlp.ipynb                    # QNN 用于自然语言处理
```

## 脚本及样例代码

```python
import sys
import mindspore as ms
import numpy as np

from qusmoke.gates import H, X, Y, Z, RX, RY, RZ, CNOT, ZZ, SWAP, U1, U2, U3
from qusmoke.circuit import Circuit

path = '../'
sys.path.append(path)                                # 添加自主开发的量子模拟器代码所在路径
ms.set_context(mode=ms.PYNATIVE_MODE)                # 使用动态图

# 构建一个基本的线路，线路中的每个参数门均为 mindspore.nn.Cell 的子类

circ = Circuit([
    H(0),
    X(0, [2,3]),
    Y(1),
    Z(1, 2),
    RX('param').on(3),
    RY(1.0).on(2, 3),
    ZZ(2.0).on((0, 1)),
    SWAP((1, 3)),
    U3(1.0, 2.0, 3.0).on(0),
    U3(1.0, 2.0, 3.0).on(1),
    U3(1.0, 2.0, 3.0).on(2),
    U3(1.0, 2.0, 3.0).on(3),
    X(1, [0, 2, 3]),
    RY(2.0).on(3, [0, 1, 2])
])

print(circ)
```

输出如下：

```log
Circuit<
  (gates): SequentialCell<
    (0): H<0>
    (1): X<0, [2, 3]>
    (2): Y<1>
    (3): Z<1, 2>
    (4): RX(param)<3>
    (5): RY(_param_)<2, 3>
    (6): ZZ(_param_)<(0, 1)>
    (7): SWAP<(1, 3)>
    (8): U3(u3_params)<0>
    (9): U3(u3_params)<1>
    (10): U3(u3_params)<2>
    (11): U3(u3_params)<3>
    (12): X<1, [0, 2, 3]>
    (13): RY(_param_)<3, [0, 1, 2]>
    >
  >
```

支持线路与量子门或其他线路相加：

```python
circ += RZ('param_z').on(3)
circ += [X(2), Y(0, 3)]
circ += Circuit([Z(1), Z(2)])

print(circ)
```

打印线路信息：

```python
circ.summary()
```

输出如下：

```log
=================================================Circuit Summary=================================================
|Total number of gates  : 19.                                                                                   |
|Parameter gates        : 9.                                                                                    |
|with 9 parameters are  :                                                                                       |
|param, _param_, _param_, u3_params, u3_params, u3_params, u3_params, _param_, param_z                        . |
|Number qubit of circuit: 4                                                                                     |
=================================================================================================================
```

获取量子态：

```python
circ.get_qs()
```

输出如下：

```python
array([ 0.08451916-0.05577248j,  0.16271482+0.0295833j ,
       -0.00641707-0.04917871j,  0.17163612-0.24420995j,
        0.16867927+0.05835877j,  0.13320881+0.259294j  ,
        0.13453737+0.00128568j,  0.15733941+0.1533909j ,
        0.14758667+0.01421016j,  0.16689745+0.08064345j,
        0.03495108+0.08881146j,  0.07131908-0.05617252j,
        0.28291166-0.45243284j,  0.06628986+0.31992695j,
        0.09377108-0.3911432j ,  0.12404367+0.21276072j], dtype=complex64)
```

**说明**：其他使用量子线路实现求解 MaxCut 问题、鸢尾花分类和NLP问题等，见 `examples/` 目录下相关代码。

## 展望

- 由于 Mindspore 目前自动微分不支持 8 维度以上张量，因此开发的量子模拟器目前最多支持 8 个量子比特的线路。
- 目前机器训练数据不支持批并行处理，但可以利用 Mindspore 的灵活性扩展。

## 参考 GIT 项目

[1] [昇思MindSpore](https://gitee.com/mindspore/mindspore)

[2] [Minspore/mindquantum](https://gitee.com/mindspore/mindquantum)

内容来源：https://gitee.com/forcekeng/qusmoke
