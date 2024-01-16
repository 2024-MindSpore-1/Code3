# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""train"""
import numpy as np
import datetime
import os
import mindspore as ms
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import Tensor
from mindspore import amp, nn
from mindspore import ops
from mindspore.common import dtype as mstype

from src.args import args
from src.tools.callback import EvaluateCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import get_criterion, NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, get_pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer
from mindspore import Profiler
from mindspore.train import ROC, auc
from sklearn.metrics import average_precision_score


def pad(x, target_shape):
    temp = Tensor(np.zeros(target_shape), dtype=ms.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            temp[i][j] = x[i][j]
    return temp

    
def main():

    set_seed(args.seed)
    #context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, max_call_depth=10000)
    context.set_context(enable_graph_kernel=False)
    #if args.device_target == "Ascend":
    #    context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)
    # get model and cast amp_level
    #profiler = Profiler()
    
    batch_size = args.batch_size

    net = get_model(args)
    net = amp.auto_mixed_precision(net, "O3")
    #cast_amp(net)
    #net = net.to_float(ms.float16)
    criterion = get_criterion(args)
    net_with_loss = NetWithLoss(net, criterion)
    if args.pretrained:
        get_pretrained(args, net)

    data = get_dataset(args)
    #batch_num = data.train_dataset.get_dataset_size()
    
    
    train_data_list = data.train_data
    batch_num = len(train_data_list) // batch_size
    test_data_list = data.test_data

    optimizer = get_optimizer(args, net, batch_num)

    #scaling_sens = Tensor([1024], dtype=ms.float32)
    #train_net = amp.build_train_network(net_with_loss, optimizer, level="O0")
    #train_net = nn.TrainOneStepWithLossScaleCell(net_with_loss, optimizer, scale_sense=scaling_sens)
    #train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
    train_net = get_train_one_step(args, net_with_loss, optimizer)

    metric = ROC()

    
    for epoch in range(args.epochs):
        print('========epoch=============')
        print(epoch)

        train_net.set_train(mode=True)
        
        for i in range(int(len(train_data_list) / batch_size)):

            for j in range(batch_size):
                if j == 0:
                    a_temp = train_data_list[i*batch_size+j][0]
                    pad = int(72 - a_temp.shape[0])
                    a_temp = np.pad(a_temp, ((0, pad), (0, 0)))
                    a_temp = Tensor(a_temp, dtype=ms.float32)
                    a = ops.unsqueeze(a_temp, dim=0)
                    
                    b_temp = train_data_list[i*batch_size+j][1]
                    pad = int(72 - b_temp.shape[0])
                    b_temp = np.pad(b_temp, ((0, pad), (0, pad)))
                    b_temp = Tensor(b_temp, dtype=ms.float32)
                    b = ops.unsqueeze(b_temp, dim=0)
                    
                    c_temp = train_data_list[i*batch_size+j][2]
                    pad = int(4682 - c_temp.shape[0])
                    c_temp = np.pad(c_temp, ((0, pad), (0, 0)))
                    c_temp = Tensor(c_temp, dtype=ms.float32)
                    c = ops.unsqueeze(c_temp, dim=0)

                    d = Tensor(train_data_list[i*batch_size+j][3], dtype=ms.int32)
                else:

                    a_temp = train_data_list[i*batch_size+j][0]
                    pad = int(72 - a_temp.shape[0])
                    a_temp = np.pad(a_temp, ((0, pad), (0, 0)))
                    a_temp = Tensor(a_temp, dtype=ms.float32)
                    a_temp = ops.unsqueeze(a_temp, dim=0)
                    a = ops.concat((a, a_temp))
                    
                    b_temp = train_data_list[i*batch_size+j][1]
                    pad = int(72 - b_temp.shape[0])
                    b_temp = np.pad(b_temp, ((0, pad), (0, pad)))
                    b_temp = Tensor(b_temp, dtype=ms.float32)
                    b_temp = ops.unsqueeze(b_temp, dim=0)
                    b = ops.concat((b, b_temp))
                    
                    c_temp = train_data_list[i*batch_size+j][2]
                    pad = int(4682 - c_temp.shape[0])
                    c_temp = np.pad(c_temp, ((0, pad), (0, 0)))
                    c_temp = Tensor(c_temp, dtype=ms.float32)
                    c_temp = ops.unsqueeze(c_temp, dim=0)
                    c = ops.concat((c, c_temp))
                    
                    d = ops.concat((d, Tensor(train_data_list[i*batch_size+j][3], dtype=ms.int32)))
            loss = train_net(a, b, c, d)
            print(loss)
            #profiler.analyse()

        net.set_train(False)
        metric.clear()
        for i in range(int(len(test_data_list) / batch_size)):
            for j in range(batch_size):
                if j == 0:
                    a_temp = train_data_list[i*batch_size+j][0]
                    pad = int(72 - a_temp.shape[0])
                    a_temp = np.pad(a_temp, ((0, pad), (0, 0)))
                    a_temp = Tensor(a_temp, dtype=ms.float32)
                    a = ops.unsqueeze(a_temp, dim=0)
                    
                    b_temp = train_data_list[i*batch_size+j][1]
                    pad = int(72 - b_temp.shape[0])
                    b_temp = np.pad(b_temp, ((0, pad), (0, pad)))
                    b_temp = Tensor(b_temp, dtype=ms.float32)
                    b = ops.unsqueeze(b_temp, dim=0)
                    
                    c_temp = train_data_list[i*batch_size+j][2]
                    pad = int(4682 - c_temp.shape[0])
                    c_temp = np.pad(c_temp, ((0, pad), (0, 0)))
                    c_temp = Tensor(c_temp, dtype=ms.float32)
                    c = ops.unsqueeze(c_temp, dim=0)

                    d = Tensor(train_data_list[i*batch_size+j][3], dtype=ms.int32)
                else:

                    a_temp = train_data_list[i*batch_size+j][0]
                    pad = int(72 - a_temp.shape[0])
                    a_temp = np.pad(a_temp, ((0, pad), (0, 0)))
                    a_temp = Tensor(a_temp, dtype=ms.float32)
                    a_temp = ops.unsqueeze(a_temp, dim=0)
                    a = ops.concat((a, a_temp))
                    
                    b_temp = train_data_list[i*batch_size+j][1]
                    pad = int(72 - b_temp.shape[0])
                    b_temp = np.pad(b_temp, ((0, pad), (0, pad)))
                    b_temp = Tensor(b_temp, dtype=ms.float32)
                    b_temp = ops.unsqueeze(b_temp, dim=0)
                    b = ops.concat((b, b_temp))
                    
                    c_temp = train_data_list[i*batch_size+j][2]
                    pad = int(4682 - c_temp.shape[0])
                    c_temp = np.pad(c_temp, ((0, pad), (0, 0)))
                    c_temp = Tensor(c_temp, dtype=ms.float32)
                    c_temp = ops.unsqueeze(c_temp, dim=0)
                    c = ops.concat((c, c_temp))
                    
                    d = ops.concat((d, Tensor(train_data_list[i*batch_size+j][3], dtype=ms.int32)))
            
            if i == 0:
                predict_batch = ops.softmax(net(a, b, c)).permute(1, 0)[1]
                d_batch = d
            else:
                predict_batch = ops.concat((predict_batch, ops.softmax(net(a, b, c)).permute(1, 0)[1]))
                d_batch = ops.concat((d_batch, d))
        print('===========predict_batch&d_batch============')
        print(predict_batch)
        print(predict_batch.shape)
        print(d_batch)
        print(d_batch.shape)
        metric.update(predict_batch, d_batch)    
        fpr, tpr, thresholds = metric.eval()
        auc_score = auc(fpr, tpr)
        print('===========auc_score==============')
        print(auc_score)
        predict_batch_numpy = predict_batch.asnumpy()
        d_batch_numpy = d_batch.asnumpy()
        prc_score = average_precision_score(d_batch_numpy, predict_batch_numpy)
        print('==========prc_score=============')
        print(prc_score)
        # max_a_test: 70  max_a_train:72
        # max_b_test: 70  max_b_train:72
        # max_a_test: 4682  max_c_train:4682
    
    exit(0)
    
    


if __name__ == '__main__':
    main()
