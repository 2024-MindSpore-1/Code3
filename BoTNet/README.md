# 目录

# BoTNet描述

## 概述

>BoTNet为UC Berkeley和谷歌团队合作发布在2021CVPR的文章，Transformer的一作Ashish Vaswani也在本文作者当中。BoTNet 原始论文为 Bottleneck Transformers for Visual Recognition。我们首先看一下 CV 领域 Self-Attention Model 分类，可分为纯 Attention Model 包括 ViT，此外则是 CNN + Attention。为什么要把 CNN 与 Attention 进行融合呢？其中一大原因就是因为纯 ViT 类型的结构对于输入尺寸特别敏感，不能变。
BoTNet 其实就是把 ResNet50 等使用的Bottleneck中间的3×3卷积给替换成了Multi-Head Self-Attention，文中缩写为 MHSA。
## 论文
[Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf)
## 参考代码
[botnet](https://github.com/BIGBALLON/distribuuuu)


## 模型架构
![BoTNet结构图](./images/botnet.png)

## 数据集

> 提供你所使用的数据信息，检查数据版权，通常情况下你需要提供下载数据的链接，数据集的目录结构，数据集大小等信息
使用的数据集：imagenet-1K, [下载地址](https://openi.pcl.ac.cn/Open_Dataset/imagenet/datasets)

数据集大小：共1000个类、224*224彩色图像

训练集：共1,281,167张图像

测试集：共50,000张图像

数据格式：JPEG


### 数据集组织方式
```bash
 └─imagenet
   ├─train                 # 训练数据集
   └─val                   # 评估数据集
```



# 特性

## 混合精度

采用[混合精度](https://www.mindspore.cn/tutorials/zh-CN/master/advanced/mixed_precision.html)的训练方法使用支持单精度和半精度数据来提高深度学习神经网络的训练速度，同时保持单精度训练所能达到的网络精度。混合精度训练提高计算速度、减少内存使用的同时，支持在特定硬件上训练更大的模型或实现更大批次的训练。
以FP16算子为例，如果输入数据类型为FP32，MindSpore后台会自动降低精度来处理数据。用户可打开INFO日志，搜索“reduce precision”查看精度降低的算子。

# 环境要求

- 硬件（Ascend/GPU）
    - 使用Ascend或GPU处理器来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```bash
  # 运行分布式训练示例
  mpirun -n 4 python train.py --config ./configs/botnet/botnet_50_ascend.yaml  --dataset_path /path/imagenet  --ckpt_save_dir /path/ckpt_save_dir  --batch_size 256 --distribute True


  # 运行评估示例
  python3 eval.py  --config ./configs/botnet/botnet_50_ascend.yaml --dataset_path /path/imagenet --ckpt_path /path/ckpt 

  ```


## 脚本说明

### 脚本和样例代码


```bash
├── README.md                    // 自述文件
├── config.py                     // 配置文件
├── configs                       // 配置目录
│   ├── README.md                 // 配置自述文件
│   └── botnet
│       └── botnet_50_ascend.yaml // botnet_50模型的配置文件
├── infer.py                      // 推断代码
├── mindcv
│   ├── data
│   │   ├── auto_augment.py       // 数据增强模块
│   │   ├── constants.py          // 常量定义
│   │   ├── dataset_download.py   // 数据集下载模块
│   │   ├── dataset_factory.py    // 数据集工厂模块
│   │   ├── distributed_sampler.py// 分布式采样器模块
│   │   ├── loader.py             // 数据加载模块
│   │   ├── mixup.py              // Mixup模块
│   │   └── transforms_factory.py // 数据预处理模块
│   ├── loss
│   │   ├── asymmetric.py         // 不对称损失模块
│   │   ├── binary_cross_entropy_smooth.py   // 平滑二值交叉熵损失模块
│   │   ├── cross_entropy_smooth.py           // 平滑交叉熵损失模块
│   │   ├── jsd.py                // Jensen-Shannon距离损失模块
│   │   └── loss_factory.py       // 损失函数工厂模块
│   ├── models
│   │   ├── features.py           // 网络特征模块
│   │   ├── botnet.py               // BotNet模型定义
│   │   ├── helpers.py            // 构建网络模块
│   │   ├── model_factory.py      // 构建网络模块
│   │   ├── registry.py           // 网络注册模块
│   │   └── layers
│   │       ├── activation.py     // 激活函数模块
│   │       ├── compatibility.py  // 兼容性模块
│   │       ├── conv_norm_act.py  // 卷积、归一化和激活模块
│   │       ├── drop_path.py      // DropPath模块
│   │       ├── helpers.py        // 模型助手函数模块
│   │       ├── identity.py       // Identity模块
│   │       ├── mlp.py            // MLP模块
│   │       ├── patch_embed.py    // Patch Embedding模块
│   │       ├── pooling.py        // 池化模块
│   │       ├── selective_kernel.py // 选择性卷积核模块
│   │       └── squeeze_excite.py // Squeeze-and-Excitation模块
│   ├── optim
│   │   ├── adamw.py              // AdamW优化器模块
│   │   ├── adan.py               // Adaptive Alpha Network优化器模块
│   │   ├── lion.py               // Lion优化器模块
│   │   ├── nadam.py              // NAdam优化器模块
│   │   └── optim_factory.py      // 优化器工厂模块
│   ├── scheduler
│   │   ├── dynamic_lr.py         // 动态学习率调度器模块
│   │   └── scheduler_factory.py  // 调度器工厂模块
│   ├── utils
│   │   ├── amp.py                // Automatic Mixed Precision模块
│   │   ├── callbacks.py          // 回调函数模块
│   │   ├── checkpoint_manager.py // 检查点管理器模块
│   │   ├── download.py           // 下载工具模块
│   │   ├── logger.py                 // 日志记录器模块
│   │   ├── path.py                    // 路径工具模块
│   │   ├── random.py                  // 随机工具模块
│   │   ├── reduce_manager.py          // 分布式训练过程中的梯度平均工具模块
│   │   ├── train_step.py              // 训练步骤模块
│   │   └── trainer_factory.py         // 训练器工厂模块
│   └── version.py                     // 版本信息模块
├── network_test.py                    // 网络测试代码
├── openi.py                           // Open平台数据模块
├── images
│   └── botnet.png                // BoTNet结构
├── requirements
│   ├── dev.txt                        // 开发环境依赖包列表
│   └── docs.txt                       // 文档生成依赖包列表
├── requirements.txt                   // 依赖包列表
├── train.py                           // 训练代码
├── train_with_func.py                 // 带有函数的训练代码
├── validate.py                        // 验证代码
└── validate_with_func.py              // 带有函数的验证代码

```
### 脚本参数

> 注解模型中的每个参数，特别是`config.py`中的参数，如有多个配置文件，请注解每一份配置文件的参数

在config.py中可以同时配置训练参数和评估参数。

- 配置ImageNet数据集。

```yaml
# system
mode: 0
distribute: True
num_parallel_workers: 32
val_while_train: True
val_interval: 1

# dataset
dataset: "imagenet"
data_dir: "/path/to/imagenet"
shuffle: True
dataset_download: False
batch_size: 128
drop_remainder: True

# augmentation
image_resize: 224
scale: [0.08, 1.0]
ratio: [0.75, 1.333]
hflip: 0.5
mixup: 0.8
cutmix: 1.0
cutmix_prob: 1.0
crop_pct: 0.875
interpolation: "bilinear"

# model
model: "botnet50"
num_classes: 1000
pretrained: False
ckpt_path: ""
keep_checkpoint_max: 30
ckpt_save_policy: "top_k"
ckpt_save_dir: "./ckpt"
epoch_size: 300
dataset_sink_mode: True
amp_level: "O2"

# loss
loss: "CE"
label_smoothing: 0.1

# lr scheduler
scheduler: "cosine_decay"
min_lr: 0.00001
lr: 0.001
warmup_epochs: 20
decay_epochs: 280

# optimizer
opt: "adamw"
filter_bias_and_bn: True
momentum: 0.9
loss_scale: 16777216.0
weight_decay: 0.05
loss_scale_type: dynamic
drop_overflow_update: True
use_nesterov: False

# train
clip_grad: True
clip_value: 10.
drop_path_rate: 0.
```
更多配置细节请参考脚本`./configs/botnet/botnet_50_ascend.yaml`。
## 训练过程

> 提供训练信息，区别于quick start，此部分需要提供除用法外的日志等详细信息

### 训练

- 启智平台智算平台Ascend NPU环境运行

参数设置
| 参数名字 | 参数 |
|---|---|
|镜像｜mindspore_1.10.1_train|
|启动文件|train.py|
|数据集|imagenet-1K|
|运行参数|👇|
|ckpt_save_dir|/cache/output/ckpt/|
|distribute|True|
|config|configs/botnet/botnet_50_ascend.yaml|
|batch_size|1024|
|资源规格| NPU: 4*Ascend 910|



训练checkpoint将被保存在智算平台的下载页面中，你可以从智算平台的日志窗口获取训练结果

```bash
[2023-09-19 06:44:00] mindcv.utils.callbacks INFO - Total time since last epoch: 265.295036(train: 250.707794, val: 13.938389)s, ETA: 530.590072s
[2023-09-19 06:44:00] mindcv.utils.callbacks INFO - --------------------------------------------------------------------------------
[2023-09-19 06:48:11] mindcv.utils.callbacks INFO - Epoch: [299/300], batch: [1251/1251], loss: 2.387096, lr: 0.000010, time: 250.705906s
[2023-09-19 06:48:25] mindcv.utils.callbacks INFO - Validation Top_1_Accuracy: 78.0960%, Top_5_Accuracy: 93.6040%, time: 14.144758s
[2023-09-19 06:48:26] mindcv.utils.callbacks INFO - Saving model to ./ckpt_1/botnet50-299_1251.ckpt
[2023-09-19 06:48:26] mindcv.utils.checkpoint_manager INFO - Top-k accuracy checkpoints:
./ckpt_1/botnet50-290_1251.ckpt	0.7822999954223633
./ckpt_1/botnet50-293_1251.ckpt	0.7822400331497192
./ckpt_1/botnet50-295_1251.ckpt	0.7821400761604309
./ckpt_1/botnet50-287_1251.ckpt	0.7816199660301208
./ckpt_1/botnet50-292_1251.ckpt	0.7815200090408325
[2023-09-19 06:48:26] mindcv.utils.callbacks INFO - Total time since last epoch: 265.510801(train: 250.715363, val: 14.144758)s, ETA: 265.510801s
[2023-09-19 06:48:26] mindcv.utils.callbacks INFO - --------------------------------------------------------------------------------
[2023-09-19 06:52:36] mindcv.utils.callbacks INFO - Epoch: [300/300], batch: [1251/1251], loss: 2.598428, lr: 0.000010, time: 250.712668s
[2023-09-19 06:52:47] mindcv.utils.callbacks INFO - Validation Top_1_Accuracy: 78.1320%, Top_5_Accuracy: 93.5560%, time: 10.724919s
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - Saving model to ./ckpt_1/botnet50-300_1251.ckpt
[2023-09-19 06:52:48] mindcv.utils.checkpoint_manager INFO - Top-k accuracy checkpoints:
./ckpt_1/botnet50-290_1251.ckpt	0.7822999954223633
./ckpt_1/botnet50-293_1251.ckpt	0.7822400331497192
./ckpt_1/botnet50-295_1251.ckpt	0.7821400761604309
./ckpt_1/botnet50-287_1251.ckpt	0.7816199660301208
./ckpt_1/botnet50-292_1251.ckpt	0.7815200090408325
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - Total time since last epoch: 262.163316(train: 250.721346, val: 10.724919)s, ETA: 0.000000s
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - --------------------------------------------------------------------------------
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - Finish training!
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - The best validation Top_1_Accuracy is: 78.2300% at epoch 290.
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - ================================================================================
```

### 分布式训练

- 启智平台智算平台Ascend NPU环境运行

参数设置
| 参数名字 | 参数 |
|---|---|
|镜像|mindspore_1.10.1_train|
|启动文件|train.py|
|数据集|imagenet-1K|
|运行参数|👇|
|ckpt_save_dir|/cache/output/ckpt/|
|distribute|True|
|config|configs/botnet/botnet_50_ascend.yaml|
|batch_size|1024|
|资源规格| NPU: 4*Ascend 910|


训练checkpoint将被保存在智算平台的下载页面中，你可以从智算平台的日志窗口获取训练结果

```text
[2023-09-19 06:44:00] mindcv.utils.callbacks INFO - Total time since last epoch: 265.295036(train: 250.707794, val: 13.938389)s, ETA: 530.590072s
[2023-09-19 06:44:00] mindcv.utils.callbacks INFO - --------------------------------------------------------------------------------
[2023-09-19 06:48:11] mindcv.utils.callbacks INFO - Epoch: [299/300], batch: [1251/1251], loss: 2.387096, lr: 0.000010, time: 250.705906s
[2023-09-19 06:48:25] mindcv.utils.callbacks INFO - Validation Top_1_Accuracy: 78.0960%, Top_5_Accuracy: 93.6040%, time: 14.144758s
[2023-09-19 06:48:26] mindcv.utils.callbacks INFO - Saving model to ./ckpt_1/botnet50-299_1251.ckpt
[2023-09-19 06:48:26] mindcv.utils.checkpoint_manager INFO - Top-k accuracy checkpoints:
./ckpt_1/botnet50-290_1251.ckpt	0.7822999954223633
./ckpt_1/botnet50-293_1251.ckpt	0.7822400331497192
./ckpt_1/botnet50-295_1251.ckpt	0.7821400761604309
./ckpt_1/botnet50-287_1251.ckpt	0.7816199660301208
./ckpt_1/botnet50-292_1251.ckpt	0.7815200090408325
[2023-09-19 06:48:26] mindcv.utils.callbacks INFO - Total time since last epoch: 265.510801(train: 250.715363, val: 14.144758)s, ETA: 265.510801s
[2023-09-19 06:48:26] mindcv.utils.callbacks INFO - --------------------------------------------------------------------------------
[2023-09-19 06:52:36] mindcv.utils.callbacks INFO - Epoch: [300/300], batch: [1251/1251], loss: 2.598428, lr: 0.000010, time: 250.712668s
[2023-09-19 06:52:47] mindcv.utils.callbacks INFO - Validation Top_1_Accuracy: 78.1320%, Top_5_Accuracy: 93.5560%, time: 10.724919s
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - Saving model to ./ckpt_1/botnet50-300_1251.ckpt
[2023-09-19 06:52:48] mindcv.utils.checkpoint_manager INFO - Top-k accuracy checkpoints:
./ckpt_1/botnet50-290_1251.ckpt	0.7822999954223633
./ckpt_1/botnet50-293_1251.ckpt	0.7822400331497192
./ckpt_1/botnet50-295_1251.ckpt	0.7821400761604309
./ckpt_1/botnet50-287_1251.ckpt	0.7816199660301208
./ckpt_1/botnet50-292_1251.ckpt	0.7815200090408325
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - Total time since last epoch: 262.163316(train: 250.721346, val: 10.724919)s, ETA: 0.000000s
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - --------------------------------------------------------------------------------
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - Finish training!
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - The best validation Top_1_Accuracy is: 78.2300% at epoch 290.
[2023-09-19 06:52:48] mindcv.utils.callbacks INFO - ================================================================================
```


## 推理

- 使用启智平台智算网络Ascend 910进行推理任务

### 推理过程

参数设置

| 参数名字 | 参数 |
|---| --- |
|AI引擎|MindSpore_1.10.1-aarch64|
|数据集|imagenet-1K|
|启动文件|validate.py|
|运行参数|👇|
|config|configs/botnet/botnet_50_ascend.yaml|
|资源规格| NPU: 1*Ascend 910|


## 性能

### 训练性能

提供您训练性能的详细描述，例如finishing loss, throughput, checkpoint size等

你可以参考如下模板

| Parameters                 | Ascend 910                                                  | 
| -------------------------- |-------------------------------------------------------------| 
| Model Version              | BoTNet50                                                    | 
| Resource                   | Ascend 910; CPU 2.60GHz, 192cores; Memory 755G; OS Euler2.8 |  
| uploaded Date              | 11/29/2023 (month/day/year)                                 | 
| MindSpore Version          | 1.10.1                                                      | 
| Dataset                    | imagenet-1K                                                 | 
| Training Parameters        | epoch=300, batch_size=1024                                  | 
| Optimizer                  | Adamw                                                       | 
| Loss Function              | Cross Entropy                                               | 
| outputs                    | probability                                                 | 
| Loss                       | 2.598428                                                    | 
| Speed                      | 267 s/epoch（4pcs）                                           | 
| Total time                 | 22 hours                                                    | 
| Parameters (M)             | 20.9                                                        | 

### 推理性能

> 提供推理性能的详细描述，包括耗时，精度等

你可以参照如下模板

| Parameters          | Ascend                   |
| ------------------- |--------------------------|
| Model Version       | BoTNet50                 |
| Resource            | Ascend 910; OS Euler2.8  |
| Uploaded Date       | 11/29/2023 (month/day/year) |
| MindSpore Version   | 1.10.1                   |
| Dataset             | imagenet-1K              |
| batch_size          | 128                      |
| outputs             | probability              |
| Accuracy            | 78.23%                   |

## 随机情况说明

> 启智平台升级可能导致有些超参数传递需要调整！


### 贡献者

此部分根据自己的情况进行更改，填写自己的院校和邮箱

* [xuxinjie2](https://openi.pcl.ac.cn/xuxinjie2) (Zhejiang University of Technology)
* [jingyangxiang](https://openi.pcl.ac.cn/ZJUTER0126) (Zhejiang University)

## ModelZoo 主页

请浏览官方[主页](https://gitee.com/mindspore/models)。

内容来源：https://openi.pcl.ac.cn/xuxinjie2/BoTNet
