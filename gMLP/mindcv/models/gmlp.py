"""
MindSpore implementation of `gmlp`.
Refer to Bottleneck Transformers for Visual Recognition.
"""

from typing import List, Optional, Type
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from .helpers import build_model_with_cfg
from .layers.pooling import GlobalAvgPooling
from .registry import register_model
from .layers.drop_path import DropPath

__all__ = [
    "gMLPForImageClassification",
    "gmlp_B",
    "gmlp_S",
    "gmlp_Ti",

]

def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "first_conv": "conv1",
        "classifier": "classifier",
        **kwargs,
    }

default_cfgs = {
    "gmlp_Ti": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt"),
    "gmlp_S": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt"),
    "gmlp_B": _cfg(url="https://download.mindspore.cn/toolkits/mindcv/resnet/resnet18-1e65cd21.ckpt"),

}




class SpatialGatingUnit(nn.Cell):
    def __init__(self, d_ffn, seq_len):
        super().__init__()
        self.norm = nn.LayerNorm([d_ffn//2]) # LayerNorm不支持传进使用int
        self.spatial_proj = nn.Conv1d(seq_len, seq_len, kernel_size=1, has_bias=True, bias_init='ones')
        # nn.init.constant_(self.spatial_proj.bias, 1.0) # todo

    def construct(self, x):
        u, v =ops.split(x, axis=-1, output_num=2)
        v = self.norm(v)
        v = self.spatial_proj(v)
        out = u * v
        return out
    

class gMLPBlock(nn.Cell):
    def __init__(self, d_model, d_ffn, seq_len, drop_path=0.):
        super().__init__()
        self.norm = nn.LayerNorm([d_model])
        self.channel_proj1 = nn.Dense(d_model, d_ffn )
        self.channel_proj2 = nn.Dense(d_ffn // 2, d_model)
        self.sgu = SpatialGatingUnit(d_ffn, seq_len)
        self.gelu = nn.GELU(approximate=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else ops.Identity()

    def construct(self, x):
        residual = x
        x = self.norm(x)
        x = self.gelu(self.channel_proj1(x))
        x = self.sgu(x)
        x = self.channel_proj2(x)
        out = self.drop_path(x) + residual
        return out
    
class gMLP(nn.Cell):
    def __init__(self, d_model=256, d_ffn=512, seq_len=256, drop_path=0., num_layers=6):
        super().__init__()
        # self.drop_path = drop_path
        self.model = nn.SequentialCell(
            *[gMLPBlock(d_model, d_ffn, seq_len, drop_path) for _ in range(num_layers)]
        )

    def construct(self, x):
        return self.model(x)
    

def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches

class gMLPForImageClassification(gMLP):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        d_model=224,
        d_ffn=512,
        seq_len=256,
        drop_path=0.,
        num_layers=6,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__(d_model, d_ffn, seq_len, drop_path, num_layers)
        # self.drop_path = drop_path
        self.patcher = nn.Conv2d(
            in_channels, d_model, kernel_size=patch_size, stride=patch_size, has_bias=True,
        )
        self.classifier = nn.Dense(d_model, num_classes)

    def construct(self, x):
        patches = self.patcher(x)
        batch_size, num_channels, _, _ = patches.shape
        patches = ops.transpose(patches, (0, 2, 3, 1))
        patches = patches.view(batch_size, -1, num_channels)
        embedding = self.model(patches)
        embedding = embedding.mean(axis=1)
        out = self.classifier(embedding)
        return out


 
@register_model
def gmlp_B(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get gmlp_B model.
    Refer to the base class `models.gmlp` for more details.
    """

    return gMLPForImageClassification(image_size=224,
                                      d_model=512,
                                        d_ffn=3072,
                                       seq_len=14*14,
                                       drop_path=0.00,
                                     num_layers=30)

@register_model
def gmlp_S(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get gmlp_S model.
    Refer to the base 1class `models.gmlp_B` for more details.
    """

    return gMLPForImageClassification(image_size=224,
                                      d_model=256,
                                        d_ffn=1536,
                                       seq_len=14*14,
                                       drop_path=0.05,
                                     num_layers=30)


@register_model
def gmlp_Ti(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs):
    """Get gmlp_Ti model.
    Refer to the base class `models.gmlp_B` for more details.
    """

    return gMLPForImageClassification(image_size=224,
                                      d_model=128,
                                        d_ffn=768,
                                       seq_len=14*14,
                                     num_layers=30)


if __name__ == '__main__':
    import numpy as np
    input_np = np.random.rand(8, 3, 224, 224)
    input_tensor = ms.Tensor(input_np, dtype=ms.float32)
    net = gmlp_Ti()

    # from torchkeras import summary
    # total = 0
    # for param in net.get_parameters():
    #     print(param.shape)
    #     total += param.size
    # print(f"Total Parameters: {total}")


    # output = net(input_tensor)
    # print(output.shape)

