# TransformerCPI （第一批）赛题十一：利用Mindspore实现化合物-蛋白质相互作用预测（生物）


# 模型简介

## 模型结构
TransformerCPI基于经典的Transformer架构，用于实现化合物-蛋白质相互作用预测，具体结构请参考https://github.com/lifanchen-simm/transformerCPI
## 数据集
BindingDB是一个二分类数据集，数据内容包括化合物表达式、蛋白质表达式及两者的相互作用，共39747个正样本核31218个负样本。这些数据被划分为训练集、验证集和测试集。其中，训练集包括28240个正样本和21915个负样本。验证集包括2831个正样本和2776个正样本。测试集包括2706个正样本和2802个正样本。数据集获取及预处理方式请参考https://github.com/lifanchen-simm/transformerCPI 本项目上传启智平台的数据集已经过预处理。在本项目中，没有用到验证集。注意：由于本数据集的划分经过特殊设计，使用数量更少的验证集进行训练，相比使用训练集进行训练，反而会在测试集上取得更好的结果，但这应该被避免。

## 代码仓地址
https://openi.pcl.ac.cn/ouyuwei/TransformerCPI

## 其它
暂无

# 代码目录结构及重要文件说明
```bash
├── README.md                           		# 模型说明文档
├── src                                 		
│   ├── configs                    		 
│   │   └── TransformerCPI.yaml                 # 参数配置文件
│   ├── data        
│   │   └── bindingDB.py                        # 数据集预处理
│   ├── models        
│   │   └── TransformerCPI.py                   # 模型定义
│   ├── tools        
│   │   └── criterion.py                        # 损失函数定义及模型损失计算
│   ├── trainer        
│   │   └── train_one_step.py                   # 模型参数更新
│   ├── args.py                    		        # 参数接收
└── train.py                            		# 训练脚本
```
# 自验结果
## 自验环境
硬件环境：NPU: 1*Ascend 910, CPU: 24, 显存: 32GB, 内存: 256GB
MindSpore版本：MindSpore-2.0-alpha-aarch64
## 训练超参数
arch: TransformerCPI                                              
run_modelarts: True
set: bindingDB # 数据集名称
optimizer: adamw # 优化器类别
base_lr: 0.0001 # 基础学习率
warmup_lr: 0.000001 # 学习率热身初始值
min_lr: 0.00001 # 最小学习率
lr_scheduler: cosine_lr # 学习率变换策略
warmup_length: 5 # 学习率热身轮数
amp_level: O3 # 混合精度级别
beta: [ 0.9, 0.999 ] # 优化器一阶、二阶梯度系数
clip_global_norm_value: 5. # 全局梯度裁剪范数
is_dynamic_loss_scale: True # 是否为动态的损失缩放
epochs: 80 # 训练轮数
cooldown_epochs: 5 # 学习率冷却稀疏
weight_decay: 0.05 # 权重衰减系数
momentum: 0.9 # 优化器动量
batch_size: 32 # 单卡批次大小
drop_path_rate: 0.1 # drop path概率
num_parallel_workers: 16 # 数据预处理线程数
device_target: Ascend # 设备选择
并行度： 单卡训练

## 训练
1、在启智平台上创建训练任务
选择AI引擎：MindSpore-2.0-alpha-aarch64 
启动文件：train.py
数据集：添加BindingDB_train.zip, BindingDB_dev.zip, BindingDB_test.zip
增加运行参数: --config /cache/user-job-dir/V0001/src/configs/TransformerCPI.yaml
规格：NPU: 1*Ascend 910, CPU: 24, 显存: 32GB, 内存: 256GB
计算节点数：1
新建任务提交训练。
2、在启智平台上进行调试
修改TransformerCPI.yaml文件中run_modelarts: False
执行命令 python train.py --config ./src/configs/TransformerCPI.yaml

## 训练精度结果
AUC: 0.9629626115732589
PRC: 0.9683402027261734


# 参考资料

## 参考论文
TransformerCPI: improving compound–protein interaction prediction by sequence-based deep learning with self-attention mechanism and label reversal experiment
## 参考git项目
https://github.com/lifanchen-simm/transformerCPI 


本文档来自于链接：https://openi.pcl.ac.cn/ouyuwei/TransformerCPI

