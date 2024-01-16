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
"""TrainOneStep"""
import datetime
import mindspore
import mindspore.nn as nn
from mindspore import ops
from mindspore.common import RowTensor
from mindspore.ops import composite as C
from mindspore.ops import functional as F
from mindspore.ops import operations as P

_grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()


@_grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * F.cast(reciprocal(scale), F.dtype(grad))


@_grad_scale.register("Tensor", "RowTensor")
def tensor_grad_scale_row_tensor(scale, grad):
    return RowTensor(grad.indices,
                     grad.values * F.cast(reciprocal(scale), F.dtype(grad.values)),
                     grad.dense_shape)


_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()


class TrainOneStep(nn.TrainOneStepWithLossScaleCell):
    """TrainOneStepWithEMA"""

    def __init__(self, network, optimizer, scale_sense=1.0, clip_global_norm_value=5.):
        super(TrainOneStep, self).__init__(network, optimizer, scale_sense)
        self.print = P.Print()
        self.clip_global_norm_value = clip_global_norm_value

    def construct(self, *inputs):
        """construct"""
        weights = self.weights

        loss = self.network(*inputs)
        
        scaling_sens = self.scale_sense
        
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        scaling_sens_filled = C.ones_like(loss) * F.cast(scaling_sens, F.dtype(loss))
 
        grads = self.grad(self.network, weights)(*inputs, scaling_sens_filled)
        grads = self.hyper_map(F.partial(_grad_scale, scaling_sens), grads)
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        # get the overflow buffer
        cond = self.get_overflow_status(status, grads)
        overflow = self.process_loss_scale(cond)
        # if there is no overflow, do optimize
        if not overflow:
            if self.clip_global_norm_value > 0:
                grads = ops.clip_by_global_norm(grads, self.clip_global_norm_value)
                grads = self.grad_reducer(grads)
            loss = F.depend(loss, self.optimizer(grads))
        else:
            self.print("=============Over Flow, skipping=============")
        return loss
