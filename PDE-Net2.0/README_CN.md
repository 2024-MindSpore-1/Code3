简体中文|[English](README.md)

# PDE-Net 2.0 求解 2D Burgers 方程

## 概述:
在PDE-Net的基础上，PDE-Net 2.0加入了symbolic network，以及针对对流项优化的pseudo-upwind等，详见论文 https://arxiv.org/pdf/1812.04426.pdf.
本项目基于Mindspore 1.10.1 框架实现了PDE-Net 2.0, 并训练求解二维Burgers方程。

* ### 训练后的 PDE-Net 2.0 对应表达式
    >```
    > =============== Current Expression ===============
    > derivative of u: -0.984473*u_x0_y0*u_x1_y0 - 0.00505433*u_x0_y0 - 0.984609*u_x0_y1*v_x0_y0 + 0.0501545*u_x0_y2 + 0.0506472*u_x2_y0
    > derivative of v: -0.983063*u_x0_y0*v_x1_y0 - 0.984026*v_x0_y0*v_x0_y1 + 0.0506355*v_x0_y2 + 0.0510178*v_x2_y0
    >```
  
* ### PDE-Net 2.0 预测结果与数值计算结果比较
    * ![comparison](images/comparison_1.png)

* ### L2 相对误差
    * ![relative_l2_error](images/relative_error_2.png)

## 快速开始
* ### 设置 部署设备(可选：CPU, GPU, Ascend)
    >```
    > export DEVICE=CPU
    >```
* ### 快速测试 1: 已经训练好的ckpt文件在 `./checkpoints` 目录下，文件名为 `pde_net_step20_epoch200.ckpt`. 如果想通过该权重文件测试，可以在终端调用 `quick_test.py` 脚本。
    >```
    > python quick_test.py --config_file_path ./config.yaml --device_target ${DEVICE} --device_id 0 --mode PYNATIVE --save_graphs False --save_graphs_path ./graphs
    >```

* ### 快速测试 1.1: 或者在终端进入`./scripts`目录，调用 `quick_test.sh` shell 脚本
    >```
    > cd {PATH}/PDENet/scripts
    > bash quick_test.sh
    >```

* ### 训练方式 1: 在终端调用 `pretrain.py` and `train.py` 脚本
    >```
    > python pretrain.py --config_file_path ./config.yaml --device_target ${DEVICE} --device_id 0 --mode PYNATIVE --save_graphs False --save_graphs_path ./graphs && python train.py --config_file_path ./config.yaml --device_target Ascend --device_id 0 --mode PYNATIVE --save_graphs False --save_graphs_path ./graphs
    >```
  
* ### 训练方式 1.1: 在终端进入`./scripts`目录，调用 `train.sh` shell 脚本
    >```
    > cd {PATH}/PDENet/scripts
    > bash train.sh
    >```

* ### 训练方式 2: 运行 Jupyter Notebook
    您可以使用 [Pretrain_中文版](poly_pdenet_pretrain_CN.ipynb) 或 [Pretrain_英文版](poly_pdenet_pretrain.ipynb) Jupyter Notebook 逐行运行预训练;
    在预训练结束后,您可以使用 [Train_中文版](poly_pdenet_train_CN.ipynb) 或 [Train_英文版](poly_pdenet_train.ipynb) Jupyter Notebook 逐行运行训练和测试代码。

* ### Requirements:
    详见[requirement](requirements.txt)

## 贡献者
* ### 止三
* ### 电子邮箱: 762598802@qq.com