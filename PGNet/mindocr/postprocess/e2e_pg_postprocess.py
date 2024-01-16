from functools import partial
from typing import Tuple, Union

import numpy as np

from mindspore import Tensor

from .e2e_utils.extract_textpoint_fast import generate_pivot_list_fast
from .e2e_utils.extract_textpoint_fast import restore_poly as restore_poly_fast
from .e2e_utils.extract_textpoint_slow import generate_pivot_list_slow
from .e2e_utils.extract_textpoint_slow import restore_poly as restore_poly_slow

__all__ = ["PGPostprocess"]


class PGPostprocess:
    def __init__(
        self,
        character_dict_path: str,
        dataset: str,
        score_thresh: float,
        mode: str,  # two different post-process mode
        point_gather_mode: str = None,
    ):
        self.char_list = [line.rstrip("\n\r") for line in open(character_dict_path, "r")]
        self.dataset = dataset.lower()
        self.score_thresh = score_thresh
        self.mode = mode.lower()

        if mode == "fast":
            self.pivot_generator = partial(
                generate_pivot_list_fast, score_thresh=self.score_thresh, point_gather_mode=point_gather_mode.lower()
            )
            self.poly_restorer = partial(restore_poly_fast, dataset=self.dataset)
        else:
            self.pivot_generator = partial(
                generate_pivot_list_slow, score_thresh=self.score_thresh, is_curved=self.dataset == "totaltext"
            )
            self.poly_restorer = partial(restore_poly_slow, dataset=self.dataset)

    def __call__(
        self, preds: Tuple[Union[Tensor, np.ndarray]], shape_list: Union[Tensor, np.ndarray], **kwargs
    ) -> dict:
        preds = list(preds)
        for i, item in enumerate(preds):
            if isinstance(item, Tensor):
                preds[i] = item.asnumpy()
        if isinstance(shape_list, Tensor):
            shape_list = shape_list.asnumpy()

        polys, scores, texts = [], [], []
        for f_score, f_border, f_char, f_direction in zip(*preds):
            all_tcls, all_scores, all_labels = self.pivot_generator(f_score, f_char, f_direction)

            all_texts = ["".join([self.char_list[i] for i in code]) for code in all_labels]

            keep_polys, keep_scores, keep_texts = self.poly_restorer(
                all_tcls, all_scores, all_texts, f_border, shape_list[0]
            )

            polys.append(keep_polys)
            scores.append(keep_scores)
            texts.append(keep_texts)

        return {"polys": polys, "scores": scores, "texts": texts}
