from mindspore import nn

from ._registry import register_backbone, register_backbone_class

__all__ = ["Identity", "identity"]

@register_backbone_class
class Identity(nn.Cell):
    def __init__(self, **kwargs):
        super().__init__()
        self.out_channels = [3]

    def construct(self, x):
        return x
    
@register_backbone
def identity(**kwargs):
    model = Identity()
    return model