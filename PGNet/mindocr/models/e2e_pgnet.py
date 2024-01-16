from ._registry import register_model
from .base_model import BaseModel
# from .backbones.mindcv_models.utils import load_pretrained

__all__ = ['PGNet', 'pgnet_resnet50']


# default_cfgs = {
#     'pgnet_resnet50': _cfg(
#         url='https://download.mindspore.cn/toolkits/mindocr/pgnet/pgnet_resnet50_???.ckpt'),
# }


def dummy(*args, **kwargs):
    return


class PGNet(BaseModel):
    """"""
    def __init__(self, config):
        super().__init__(config)


@register_model
def pgnet_resnet50(pretrained=False, pytorch_like=False, **kwargs):
    model_config = {
        "backbone": {
            'name': 'e2e_resnet50',
            'pretrained': False,
            'pytorch_like': pytorch_like
        },
        "neck": {
            "name": 'PGFPN',
            'pytorch_like': pytorch_like
        },
        "head": {
            "name": 'PGHead',
            "num_classes": 37
        }
    }
    model = PGNet(model_config)

    # load pretrained weights
    # if pretrained:
        # print('no pretrain choice, train from scratch')
        # raise NotImplementedError()
        # default_cfg = default_cfgs['pgnet_resnet50']
        # load_pretrained(model, default_cfg)

    return model
