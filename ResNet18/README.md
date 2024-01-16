# 目录

<!-- TOC -->

- [目录](#目录)
- [项目修改部分](#项目修改部分)
- [异常原因](#异常原因)
- [解决方案](#解决方案)
<!-- /TOC -->


# 基于MindSpore实现Python脚本加密算法，支持图模式训练解决方案

## 题目介绍
问题描述
模型的Python脚本包含了模型结构设计、模型训练算法等关键信息，使用模型脚本在不安全的环境中推理和训练存在被窃取的风险，需要设计有效的方案保护模型的Python脚本的机密性。目前业界有多种Python文件加密工具，其中包括pyarmour，但是该工具仅仅支持MindSpore动态图模型下的推理和训练。参赛者需要基于已有的开源工具和MindSpore源码，实现能支持图模式（Graph_mode）下的加密训练方案。
参考资料
https://zhuanlan.zhihu.com/p/609709232
https://zhuanlan.zhihu.com/p/605818200
评判标准
加密的MindSpore模型（ResNet50）的Python脚本可以在图模式下进行训练。
评分规则
采取竞速规则，即最先按要求完成赛题并通过验证的队伍赢得奖金。

## 项目说明
本项目基于Resnet,实现在对python脚本加密后，依旧能够进行图模式训练的方案
项目结构及项目内其他信息，请参考https://openi.pcl.ac.cn/dexg/Resnet/src/branch/master/README_CN.md

## 项目修改部分
文件结构：
大部分resnet一致，修改的部分如下：
config/resnet50_cifar10_config.yaml:修改了配置信息，包含dataset路径等
function_attributes.py：新增，用于添加函数装饰器，用于更新sources.py的函数
src/resnet.py：添加了函数装饰器及相关import
src_bak/sources.py:新增，储存网络源码信息

## 异常原因
python脚本被加密后，不能通过c++底层函数的图模式编译

异常位置
https://gitee.com/mindspore/mindspore/blob/r2.0/mindspore/python/mindspore/common/api.py

```
from mindspore._c_expression import GraphExecutor_
...
self._graph_executor = GraphExecutor_.get_instance()
...
if jit_config_dict:
    self._graph_executor.set_jit_config(jit_config_dict)
result = self._graph_executor.compile(obj, args, kwargs, phase, self._use_vm_mode())  #触发异常，编译不通过
obj.compile_cache.add(phase)
```

经过查找，要通过这部分编译，需要提供网络结构的construct方法的源代码

## 解决方案

### 一、提前获取construct函数的结构，并声明
例如：[sources.py](https://openi.pcl.ac.cn/dexg/Resnet/src/branch/master/src/sources.py)
```
ResNet_source=(['    def construct(self, x):\n', '        if self.use_se:\n', '            x = self.conv1_0(x)\n', '            x = self.bn1_0(x)\n', '            x = self.relu(x)\n', '            x = self.conv1_1(x)\n', '            x = self.bn1_1(x)\n', '            x = self.relu(x)\n', '            x = self.conv1_2(x)\n', '        else:\n', '            x = self.conv1(x)\n', '        x = self.bn1(x)\n', '        x = self.relu(x)\n', '        if self.res_base:\n', '            x = self.pad(x)\n', '        c1 = self.maxpool(x)\n', '\n', '        c2 = self.layer1(c1)\n', '        c3 = self.layer2(c2)\n', '        c4 = self.layer3(c3)\n', '        c5 = self.layer4(c4)\n', '\n', '        out = self.mean(c5, (2, 3))\n', '        out = self.flatten(out)\n', '        out = self.end_point(out)\n', '\n', '        return out\n'],475)
```

这部分代码可以通过一下步骤自动生成
#### 1、创建一个sources.py文件
文件内进行变量声明，例如
```
ResNet_source=None
```
#### 2、在网络construct函数前添加装饰器
```
import function_attributes as fa
from src.sources import *
...
    @fa.set_attribute('source', ResNet_source)#注意，这里的ResNet_source变量即为步骤1内的ResNet_source=None声明的变量
    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.use_se and self.stride != 1:
            out = self.e2(out)
...
```
#### 3、运行这段代码
```
import function_attributes as fa
import src_bak.resnet as resnet
#需要在ResNet和ResidualBlock的construct函数前添加装饰器，装饰器函数传递的参数需要先在sources.py内声明

fa.creat_source_file([resnet.ResNet,resnet.ResidualBlock],"/code/src_bak/sources.py")
print("source传递源码参数准备完毕！")
```
creat_source_file传递的第一个参数为需要处理的网络类，第二个参数为sources.py的路径，
经过这一步骤后，自动生成一个新的sources.py


creat_source_file函数的代码及注释如下：
```

def get_name(stra):
    match = re.search(r',(.*?)\)', stra)  
    if match:  
        extracted_string = match.group(1)  
        #print(extracted_string)  # 输出: ResidualBlock_source 
        return extracted_string
    else:  
        raise Exception("No match found.")
            
def creat_code(net):
    lines,line_offset=inspect.getsourcelines(net.construct)#获取网络construct方法的源代码
    var = get_name(lines[0]).strip()#lines[0]其实就是字符串形式的'@fa.set_attribute('source', ResNet_source)'，这个函数的目的使用获取ResNet_source这个变量名
    strx=f"{var}=({str(lines[1:])},{str(line_offset+1)})"#以字符串的形式生成source.py的内容
    return strx
#生成sources.py文件
def creat_source_file(nets,filename="sources.py"):
    strs=[creat_code(net) for net in nets]#遍历所有网络
    with open(filename, 'w') as f:  
        for item in strs:  
            f.write("%s\n" % item)
```
### 二、加密运行

加密后，提供源码的Source.py也会被加密，而且不影响图模式下的训练

详细可以查看https://openi.pcl.ac.cn/dexg/Resnet/src/branch/master/run.ipynb 内的运行流程

### 三、运行环境

新建一个调试任务，选择启智GPU进行调试。
镜像：mindspire2.0
硬件资源：CPU
数据集：选择项目内cifar-10数据集