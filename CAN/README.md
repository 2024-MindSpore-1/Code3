# 利用MindSpore实现CAN模型(昇腾AI创新大赛2023-昇思赛道-算法创新赛题)

## 题目介绍
[**昇腾AI创新大赛2023-昇思赛道-算法创新赛题**](https://www.hiascend.com/zh/developer/contests/details/7e51ee21cd604773b5b91b974cef30be)
<br>本题目要求利用MindSpore实现CAN模型，[CAN](https://arxiv.org/abs/2207.11463) 是ECCV 2022的一篇工作，在手写数学公式识别中有出色的表现。

## 使用的数据集
CROHME14数据集是针对在线手写数学表达式识别的标准数据集。它包含了各种算术、代数和高级数学的手写公式，并使用MathML或其他标记语言进行注释。该数据集旨在评估和比较不同的HMER方法。
<br>从[百度云](https://pan.baidu.com/s/1qUVQLZh5aPT6d7-m6il6Rg) 可以下载 CROHME 数据集（downloading code: 1234）并将其放置到 ```datasets/```。

## 环境要求
**Ascend:** Ascend910
<br>**Mindspore:** Mindspore1.10.1

## 目录结构
```
│  AttDecoder_bad_case.json
│  config.yaml
│  counting_utils.py
│  dataset.py
│  inference.py
│  LICENSE
│  new.ckpt
│  README.md
│  requirements.txt
│  run.ipynb
│  train.py
│  training.py
│  utils.py
│
├─datasets
│  └─CROHME
│          14_test_images.pkl
│          14_test_labels.txt
│          16_test_images.pkl
│          16_test_labels.txt
│          19_test_images.pkl
│          19_test_labels.txt
│          train_images.pkl
│          train_labels.txt
│          words_dict.txt
│
├─models
│      attention.py
│      can.py
│      counting.py
│      decoder.py
│      densenet.py
│      infer_model.py
│      __init__.py
│
└─scripts
        run_inference.sh
        run_train.sh
```

## 快速开始
### 模型训练
```shell
python train.py --dataset CROHME
```
### 模型推理
```shell
python inference.py --dataset CROHME
```

## 评估结果
### 模型训练精度如下：

| 模型        | 精度:ExpRate |
|-----------|------------|
| CAN(ours) | 54.26%     |
| Paddle    | 51.72%     |
| 论文        | 57.00%     |
### 模型推理结果如下：
![inference_result.png](assets%2Finference_result.png)

## 参考资料
[When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2207.11463)
<br>[基于pytorch实现CAN模型](https://github.com/LBH1024/CAN)
<br>[基于PaddlePaddle实现CAN模型](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/algorithm_rec_can.md)


内容来源：https://github.com/kingkingofall/ms_maxwellnet