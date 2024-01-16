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
"""misc functions for program"""
import collections.abc
import os
from itertools import repeat

from mindspore import nn, context
from mindspore.communication.management import init, get_rank
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src import data, models
from src.data.data_utils.moxing_adapter import sync_data
from src.trainer.train_one_step import TrainOneStep


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def set_device(args):
    """Set device and ParallelMode(if device_num > 1)"""
    rank = 0
    # set context and device
    device_target = args.device_target
    device_num = int(os.environ.get("DEVICE_NUM", 1))

    if device_target == "Ascend":
        print('=====device_num======')
        print(device_num)
        print('=====os.environ["DEVICE_ID"]======')
        print(os.environ["DEVICE_ID"])
        if device_num > 1:
            context.set_context(device_id=int(os.environ["DEVICE_ID"]))
            init(backend_name='hccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            #context.set_context(device_id=args.device_id)
            pass
    elif device_target == "GPU":
        if device_num > 1:
            init(backend_name='nccl')
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        else:
            #context.set_context(device_id=args.device_id)
            pass
    else:
        pass

    return rank


def get_dataset(args, training=True):
    """"Get model according to args.set"""
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args, training)

    return dataset


def get_model(args):
    """"Get model according to args.arch"""
    print("==> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    return model


def get_pretrained(args, model):
    """"Load pretrained weights if args.pretrained is given"""
    if args.run_modelarts:
        print('Syncing data.')
        local_data_path = '/cache/weight/model.ckpt'
        sync_data(args.pretrained, local_data_path, threads=128)
        print("=> loading pretrained weights from '{}'".format(local_data_path))
        param_dict = load_checkpoint(local_data_path)
        load_param_into_net(model, param_dict)
    elif os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        param_dict = load_checkpoint(args.pretrained)
        for key, value in param_dict.copy().items():
            if 'head' in key:
                if value.shape[0] != args.num_classes:
                    print(f'==> removing {key} with shape {value.shape}')
                    param_dict.pop(key)
        load_param_into_net(model, param_dict)
    else:
        print("=> no pretrained weights found at '{}'".format(args.pretrained))


def get_train_one_step(args, net_with_loss, optimizer):
    """get_train_one_step cell"""
    if args.is_dynamic_loss_scale:
        print(f"=> Using DynamicLossScaleUpdateCell")
        scale_sense = nn.wrap.loss_scale.DynamicLossScaleUpdateCell(loss_scale_value=2 ** 24, scale_factor=2,
                                                                    scale_window=2000)
    else:
        print(f"=> Using FixedLossScaleUpdateCell, loss_scale_value:{args.loss_scale}")
        scale_sense = nn.wrap.FixedLossScaleUpdateCell(loss_scale_value=args.loss_scale)

    net_with_loss = TrainOneStep(
        net_with_loss, optimizer, scale_sense=scale_sense, clip_global_norm_value=args.clip_global_norm_value)

    return net_with_loss
