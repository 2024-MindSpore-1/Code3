from typing import Dict, List

import numpy as np

from ..core import Model, ShapeType
from ..data_process import build_postprocess, build_preprocess, cv_utils, gear_utils
from .infer_base import InferBase


class TextEnd2End(InferBase):
    def __init__(self, args):
        super().__init__(args)

    def _init_preprocess(self):
        self.preprocess_ops = build_preprocess(self.args.e2e_config_path, self.requires_gear_hw)

    def _init_model(self):
        self.model = Model(
            backend=self.args.backend,
            device=self.args.device,
            model_path=self.args.e2e_model_path,
            device_id=self.args.device_id,
        )

        shape_type, shape_value = self.model.get_shape_details()

        # Only check inputs[0] currently
        shape_value = shape_value[0]

        # check batch_size
        # assuming that the first dim is batch size, make sure that the batch_size must support 1
        batch_size = shape_value[0]
        if (batch_size not in [1, -1]) and (
            isinstance(batch_size, (tuple, list)) and 1 not in batch_size  # dynamic batch size list: [1, ...]
        ):
            raise ValueError("Input batch size must support 1 for detection model.")
        batch_size = [batch_size] if not isinstance(batch_size, (tuple, list)) else batch_size

        # check h/w
        if shape_type == ShapeType.DYNAMIC_SHAPE:
            # without any checks, and assuming that h/w is dynamic
            # if not, may throw exceptions in model_infer for un-matched h/w
            hw_list = []
        elif shape_type == ShapeType.DYNAMIC_IMAGESIZE:  # only support NCHW
            *_, hw_list = shape_value
        else:  # static shape or dynamic batch size
            if len(shape_value) == 4:  # only support NCHW
                *_, h, w = shape_value
                hw_list = [(h, w)]
            else:
                hw_list = []  # without any checks

        self._hw_list = tuple(hw_list)
        self._bs_list = tuple(batch_size)

    def _init_postprocess(self):
        self.postprocess_ops = build_postprocess(self.args.e2e_config_path)

    def get_params(self):
        return {"e2e_batch_num": self._bs_list}

    def __call__(self, images: List[np.ndarray]) -> List:
        outputs = []
        for image in images:
            data = self.preprocess(image)
            pred = self.model_infer(data)
            polys = self.postprocess(pred, data["shape_list"])
            outputs.append(polys)
        return outputs

    def preprocess(self, image: np.ndarray) -> Dict:
        if self.requires_gear_hw:
            target_size = gear_utils.get_matched_gear_hw(cv_utils.get_hw_of_img(image), self._hw_list)
            data = self.preprocess_ops([image], target_size=target_size)
        else:
            data = self.preprocess_ops([image])
        return data

    def model_infer(self, data: Dict) -> List[np.ndarray]:
        """return
        - f_score     : [b, 1, h, w]
        - f_border    : [b, 4, h, w]
        - f_char      : [b, c, h, w]
        - f_direction : [b, 2, h, w]
        """
        preds = self.model.infer(data["net_inputs"])
        return preds

    def postprocess(self, pred: List[np.ndarray], shape_list: np.ndarray) -> List[np.ndarray]:
        """{'polys': [img0_polys, ...], 
            'scores': [img0_scores, ...],
            'texts': [img0_texts, ...]}
            虽然代码支持bs>1，但保持bs=1应该效果最好
        """
        preds = self.postprocess_ops(tuple(pred), shape_list)
        return preds
