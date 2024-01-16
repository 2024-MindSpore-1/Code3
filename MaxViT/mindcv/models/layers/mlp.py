""" MLP module w/ dropout and configurable activation layer
"""
from typing import Optional

import mindspore.common.initializer as init
from mindspore import Tensor, nn
from mindspore.common.initializer import Normal

from .compatibility import Dropout
from .helpers import to_2tuple


class Mlp(nn.Cell):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[nn.Cell] = nn.GELU,
        drop: float = 0.0,
        has_bias: bool = True
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=has_bias)
        self.act = act_layer()
        self.fc2 = nn.Dense(in_channels=hidden_features, out_channels=out_features, has_bias=has_bias)
        self.drop = Dropout(p=drop)

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class GatedMlp(nn.Cell):
    """ MLP as used in gMLP
    """

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            gate_layer=None,
            bias=True,
            drop=0.,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        self.bias = bias
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Dense(in_channels=in_features, out_channels=hidden_features, has_bias=bias[0])
        self.act = act_layer()
        self.drop1 = Dropout(p=drop_probs[0])
        if gate_layer is not None:
            assert hidden_features % 2 == 0
            self.gate = gate_layer(hidden_features)
            hidden_features = hidden_features // 2
        else:
            self.gate = nn.Identity()
        self.norm = norm_layer((hidden_features,)) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Dense(hidden_features, out_features, has_bias=bias[1])
        self.drop2 = Dropout(p=drop_probs[1])
        self.init_weights()

    def init_weights(self):
        self.fc1.weight.set_data(init.initializer(Normal(sigma=1e-6), self.fc1.weight.shape, self.fc1.weight.dtype))
        self.fc2.weight.set_data(init.initializer(Normal(sigma=1e-6), self.fc2.weight.shape, self.fc2.weight.dtype))
        if self.bias[0]:
            self.fc1.bias.set_data(init.initializer("ones", self.fc1.bias.shape, self.fc1.bias.dtype))
        if self.bias[1]:
            self.fc2.bias.set_data(init.initializer("ones", self.fc2.bias.shape, self.fc2.bias.dtype))

    def construct(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.gate(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
