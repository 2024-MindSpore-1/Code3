# **1. 模型简介**

## **1.1. 模型结构简介**

SepFormer是一个仅由[transformer](https://so.csdn.net/so/search?q=transformer&spm=1001.2101.3001.7020)组成的掩蔽网络，通过transformer的多尺度学习方法学习短期和长期依赖关系，使用DPRNN的双尺度框架，将DPRNN中的RNN替换为transformer组成的多尺度管道,DPRNN证明更好的长期建模对于提高分离性能至关重要.

模型结构

- 模型是基于学习的域掩蔽方法，有编码器、解码器和掩蔽网络组成
- 编码器：时域混合语音作为输入，它是一个一维卷积，卷积的步长对模型的性能、速度和内存有显著影响
- 掩蔽网络：编码器的输出作为输入，经过层归一化和线性层后分块，之后进入SepFormer学习短期和长期依赖关系，之后进入激活函数和线性层
- SepFormer：使用了类似DPRNN中的双尺度方法对短期和长期依赖性进行建模。对短期相关性进行建模的transformer块称为intranformer（IntraT），而对长期相关性进行建模的transformer块称为InterTransformer（InterT）
- 解码器：使用转置卷积，具有与编码器相同的步长和核大小。输入是混合声音的掩蔽和编码器输出的元素乘积
  

## **1.2. 数据集**

- libri2mix

  https://arxiv.org/pdf/2005.11262.pdf

## **1.3. 代码提交地址**

https://openi.pcl.ac.cn/longmx/sepformer

# **2. 代码目录结构说明**

```bash
sepformer
├── dataprocess
├── hparams
       ├── sepformer.yaml
├── src
│   ├──attention.py
│   ├── augment.py
│   ├── datasets.py
│   ├── dual_path.py
│   ├── linear.py
│   ├── losses.py
│   ├── normalization.py
│   ├── speech_augmentation.py
│   ├── transformer.py
├── scripts
│   ├──run_eval.sh
│   ├──run_standalone_train.sh
└── train.py
└── eval.py
└── readme.md
```
# **3. 自验结果**

## **3.1. 自验环境**

Ascend910 + MindSpore2.0.0 + Python3.8.0

## **3.2. 训练超参数**

详细训练超参数请查看./sepformer.yaml文件夹中的配置文件。

## **3.3. 训练**

### **3.3.1 如何启动训练脚本**

```bash
bash run_standalone_train.sh DATA_PATH
```

### **3.3.2 训练精度结果**

|  dataset  | sepformer |
| :-------: | :-------: |
| libri2mix |   18.94   |




# **4. 参考资料**

## **4.1. 参考论文**

https://arxiv.org/pdf/2010.13154.pdf

## **4.2. 参考git项目**

https://github.com/speechbrain


内容来源：https://openi.pcl.ac.cn/longmx/sepformer



