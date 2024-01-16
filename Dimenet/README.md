# DimenetPP （第一批）赛题十：利用Mindspore预测分子性质（化学材料）

# 模型简介
## 模型结构
Dimenet是最近提出的一种用于分子性质预测的图神经网络。DimeNet嵌入并更新原子之间的消息mji，这使它能够考虑方向信息（通过键角α（kj，ji））以及原子间距离dji，并进一步使用球面2D傅立叶贝塞尔基联合嵌入距离和角度。Dimenet++没有触及上述贡献，而是对模型架构进行了一定修改，具体请参考https://arxiv.org/pdf/2011.14115.pdf。

## 数据集
QM9为小有机分子的相关的、一致的和详尽的化学空间提供量子化学特征，该数据库可用于现有方法的基准测试，新方法的开发，如混合量子力学/机器学习，以及结构-性质关系的系统识别。

新药物和新材料的计算从头设计需要对化合物空间进行严格和公正的探索。然而，由于其大小与分子大小相结合，大量未知领域仍然存在。报告计算了由CHONF组成的134k稳定有机小分子的几何、能量、电子和热力学性质。这些分子对应于GDB-17化学宇宙中1660亿个有机分子中所有133,885个含有多达9个重原子(CONF)的物种的子集。报告了能量最小的几何，相应的谐波频率，偶极矩，极化率，以及能量，焓，和原子化的自由能。所有性质都是在量子化学的B3LYP/6-31G(2df,p)水平上计算的。此外，对于主要的化学计量，C7H10O2，在134k分子中有6095个组成异构体。在更精确的G4MP2理论水平上报告了所有这些原子化的能量、焓和自由能。因此，该数据集为相关、一致和全面的小有机分子化学空间提供了量子化学性质。该数据库可用于现有方法的基准测试，新方法的开发，如混合量子力学/机器学习，以及结构-性质关系的系统识别。

上述介绍参考https://blog.csdn.net/KPer_Yang/article/details/129105477

数据集获取请参考https://github.com/gasteigerjo/dimenet

## 代码仓地址
https://openi.pcl.ac.cn/ouyuwei/Dimenetpp

## 其它
（1）该模型存在多个输入均是动态shape，如果直接在mindspore上执行会重复编图很慢不说，一个迭代都跑不完就会内存不足。为了解决该问题，需要对输入进行填充。该模型输入很特殊，除了用于计算的数据外，还有大量索引用于区分每个位置上的数据对应第几个输入样本，简单的填充方式根本无法满足该模型要求，于是本人对填充方式以及后续计算过程进行了创新。首先指定各个输入的最大长度，然后对于用于计算的数据填充0至指定长度，对于索引数据填充之前多出来数据的索引号，然后传入模型进行计算。多出来的计算部分会导致NAN和inf，直接改成0即可，因为这部分本来就不参与梯度传播。假设batch size是32，传入模型的相当于33个数据（32个有效数据+1个无效数据）。模型的输入也包括33个数据。在计算loss的时候，只取前32个数据进行计算。通过与原方式进行loss比对，本人确认该方法执行过程是正确的。

（2）原论文的代码使用sympy库将sympy符号函数转换为tensorflow函数，但是sympy库不支持mindspore，所以在涉及这部分功能时，本人手动写出对应的函数。


# 代码目录结构及重要文件说明
```bash
├── README.md                           		# 模型说明文档
├── src                                 		
│   ├── configs                    		 
│   │   └── dimenetpp.yaml                 # 参数配置文件
│   ├── data        
│   │   └── qm9.py                        # 数据集预处理
│   ├── models        
│   │   └── dimenetpp.py                   # 模型定义
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
arch: DimenetPP  # 模型结构
run_modelarts: True
set: qm9 # 数据集名称
optimizer: adamw # 优化器类别
base_lr: 0.001 # 基础学习率
warmup_lr: 0.000001 # 学习率热身初始值
min_lr: 0.00001 # 最小学习率
lr_scheduler: cosine_lr # 学习率变换策略
warmup_length: 1 # 学习率热身轮数
amp_level: O3 # 混合精度级别
keep_bn_fp32: True # 是否保持BN为FP32运算
beta: [ 0.9, 0.999 ] # 优化器一阶、二阶梯度系数
clip_global_norm_value: 5. # 全局梯度裁剪范数
is_dynamic_loss_scale: True # 是否为动态的损失缩放
epochs: 50 # 训练轮数
cooldown_epochs: 1 # 学习率冷却稀疏
label_smoothing: 0.1 # 标签平滑稀疏
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
数据集：添加qm9_eV.zip
增加运行参数: --config /home/work/user-job-dir/code/src/configs/dimenetpp.yaml
规格：NPU: 1*Ascend 910, CPU: 24, 显存: 32GB, 内存: 256GB
计算节点数：1
新建任务提交训练。

## 训练精度结果
U0的MAE:简单地训练了50epoch（原论文训练了750个epoch）。loss降到0.2就降不动了，与原论文有较大差距。

如果要预测其它属性，在/src/data/qm9.py中修改下target



# 参考资料
## 参考论文
Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules
## 参考git项目
https://github.com/gasteigerjo/dimenet

文档内容来自：https://openi.pcl.ac.cn/ouyuwei/Dimenetpp

