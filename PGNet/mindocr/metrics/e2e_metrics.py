import os.path as osp
from functools import partial
from typing import Tuple

import numpy as np
import scipy.io as io
from shapely.geometry import Polygon

from mindspore import Tensor, dtype, nn

from mindocr.structures.polygon import area_of_intersection, iod

from ..utils.misc import AllReduce

__all__ = ["E2EMetric"]


def _safe_divide(numerator, denominator, val_if_zero_divide=0.0) -> float:
    return val_if_zero_divide if denominator == 0 else (numerator / denominator)


def sigma_calculation(det_p: Polygon, gt_p: Polygon) -> float:
    """sigma = inter_area / gt_area"""
    return _safe_divide(area_of_intersection(det_p, gt_p), gt_p.area)


def tau_calculation(det_p: Polygon, gt_p: Polygon) -> float:
    """tau = inter_area / det_area"""
    return _safe_divide(area_of_intersection(det_p, gt_p), det_p.area)


def read_mat(path: str) -> tuple:
    assert osp.exists(path), f'"{path}" not exist'
    raw_data = io.loadmat(path)
    polys, texts, ignores = [], [], []
    for info in raw_data["polygt"]:
        if info[1].shape[1] > 1:
            polys.append(Polygon(np.concatenate([info[1], info[3]]).T.astype(np.int32)))
            ignores.append("".join(info[5].tolist()) == "#")
        else:
            polys.append(None)
            ignores.append(True)
        texts.append("".join(info[4].tolist()))
    return polys, texts, ignores


class E2EEvaluator:
    def __init__(self, **kwargs):
        self.sigma_thr = 0.7
        self.tau_thr   = 0.6
        self.fsc_k = 0.8
        self.k = 2

    @staticmethod
    def filtering(gts: Tuple[tuple], dets: Tuple[tuple], threshold: float = 0.5) -> Tuple[Tuple[tuple]]:
        new_gts, new_dets = [], dets
        for gt_poly, gt_text, gt_ignore in gts:
            if not gt_ignore:
                new_gts.append((gt_poly, gt_text))
            elif gt_poly is not None:
                new_dets = []
                for det_poly, det_text in dets:
                    det_gt_iou = iod(det_poly, gt_poly)
                    if det_gt_iou <= threshold:  # 没有和无效gt匹配
                        new_dets.append((det_poly, det_text))
                dets = new_dets
        return new_gts, new_dets

    def one_to_one(
        self,
        sigma_table: np.ndarray,
        tau_table: np.ndarray,
        accumulative_recall: int,
        accumulative_precision: int,
        gt_flag: np.ndarray,
        det_flag: np.ndarray,
        gt_texts: list,
        pred_texts: list,
        num_hit: int = 0,
    ):
        match = (sigma_table > self.sigma_thr) & (tau_table > self.tau_thr)
        # one2one_candidates
        gt_ids, det_ids = np.meshgrid(np.where(match.sum(1) == 1)[0], np.where(match.sum(0) == 1)[0])
        valid = match[gt_ids, det_ids]

        num_valid = valid.sum()
        if num_valid > 0:
            accumulative_recall    += num_valid
            accumulative_precision += num_valid

            for gt_id, det_id in zip(gt_ids[valid], det_ids[valid]):
                gt_flag [ gt_id] = True
                det_flag[det_id] = True
                if pred_texts[det_id].lower() == gt_texts[gt_id].lower():
                    num_hit += 1

        return accumulative_recall, accumulative_precision, gt_flag, det_flag, num_hit

    def one_to_many(
        self,
        sigma_table: np.ndarray,
        tau_table: np.ndarray,
        accumulative_recall: int,
        accumulative_precision: int,
        gt_flag: np.ndarray,
        det_flag: np.ndarray,
        gt_texts: list,
        pred_texts: list,
        num_hit: int = 0,
    ):
        num_gt, num_det = sigma_table.shape
        for gt_id in range(num_gt):
            # skip the following if the groundtruth was matched
            if not gt_flag[gt_id] and np.sum(sigma_table[gt_id] > 0) >= self.k:
                # search for all detections that overlaps with this groundtruth
                det_ids = np.where((tau_table[gt_id] >= self.tau_thr) & ~det_flag)[0]
                num_candidates = det_ids.shape[0]

                if num_candidates == 1:
                    if  (sigma_table[gt_id, det_ids] >= self.sigma_thr) and \
                        (tau_table  [gt_id, det_ids] >= self.tau_thr):
                        # became an one-to-one case
                        accumulative_recall    += 1
                        accumulative_precision += 1

                        gt_flag [  gt_id] = True
                        det_flag[det_ids] = True

                        if pred_texts[det_ids[0]].lower() == gt_texts[gt_id].lower():
                            num_hit += 1

                elif np.sum(sigma_table[gt_id][det_ids]) >= self.sigma_thr:
                    accumulative_recall    += self.fsc_k
                    accumulative_precision += self.fsc_k * num_candidates

                    gt_flag [  gt_id] = True
                    det_flag[det_ids] = True

                    # FIXME why？
                    if pred_texts[det_ids[0]].lower() == gt_texts[gt_id].lower():
                        num_hit += 1

        return accumulative_recall, accumulative_precision, gt_flag, det_flag, num_hit

    def many_to_one(
        self,
        sigma_table: np.ndarray,
        tau_table: np.ndarray,
        accumulative_recall: int,
        accumulative_precision: int,
        gt_flag: np.ndarray,
        det_flag: np.ndarray,
        gt_texts: list,
        pred_texts: list,
        num_hit: int = 0,
    ):
        num_gt, num_det = sigma_table.shape
        for det_id in range(num_det):
            # skip the following if the detection was matched
            if not det_flag[det_id] and np.sum(tau_table[:, det_id] > 0) >= self.k:
                # search for all detections that overlaps with this groundtruth
                gt_ids = np.where((sigma_table[:, det_id] >= self.sigma_thr) & ~gt_flag)[0]  # FIXME 原paddle-ocr和tau_thr比
                num_candidates = gt_ids.shape[0]

                if num_candidates == 1:
                    if  (sigma_table[gt_ids, det_id] >= self.sigma_thr) and \
                        (tau_table  [gt_ids, det_id] >= self.tau_thr):
                        # became an one-to-one case
                        accumulative_recall    += 1
                        accumulative_precision += 1

                        gt_flag [gt_ids] = True
                        det_flag[det_id] = True

                        if pred_texts[det_id].lower() == gt_texts[gt_ids[0]].lower():
                            num_hit += 1

                elif np.sum(tau_table[gt_ids, det_id]) >= self.tau_thr:
                    accumulative_recall    += self.fsc_k * num_candidates
                    accumulative_precision += self.fsc_k

                    det_flag[det_id] = True
                    gt_flag [gt_ids] = True

                    pred_text = pred_texts[det_id]
                    for gt_id in gt_ids:
                        if pred_text.lower() == gt_texts[gt_id].lower():
                            num_hit += 1
                            break

        return accumulative_recall, accumulative_precision, gt_flag, det_flag, num_hit

    def __call__(self, total_dets_and_gts: list) -> tuple:
        total_recall, total_precision = 0.0, 0.0
        total_num_gt, total_num_det, total_num_hit = 0, 0, 0

        # for i, (gts, dets) in enumerate(total_dets_and_gts):  # DEBUG
        for gts, dets in total_dets_and_gts:
            gts, dets = self.filtering(gts, dets)

            num_gt, num_det = len(gts), len(dets)

            if num_det > 0:
                sigma_table = np.zeros((num_gt, num_det), dtype=np.float32)
                tau_table   = np.zeros((num_gt, num_det), dtype=np.float32)
                for gt_id, (gt_poly, gt_text) in enumerate(gts):
                    for det_id, (det_poly, det_text) in enumerate(dets):
                        sigma_table[gt_id, det_id] = sigma_calculation(det_poly, gt_poly)
                        tau_table  [gt_id, det_id] =   tau_calculation(det_poly, gt_poly)

                gt_texts   = [ gt[1] for  gt in  gts]
                pred_texts = [det[1] for det in dets]

                gt_flag  = np.zeros(num_gt , dtype=bool)  # whether matched
                det_flag = np.zeros(num_det, dtype=bool)

                for func in [self.one_to_one, self.one_to_many, self.many_to_one]:
                    total_recall, total_precision, gt_flag, det_flag, total_num_hit = func(
                        np.round(sigma_table, 2),
                        np.round(tau_table, 2),
                        total_recall,
                        total_precision,
                        gt_flag,
                        det_flag,
                        gt_texts,
                        pred_texts,
                        total_num_hit,
                    )

            total_num_gt += num_gt
            total_num_det += num_det

        return total_recall, total_precision, total_num_gt, total_num_det, total_num_hit


def code2text(codes, char_map, special_id: int = -1, special_char: str = "*"):
    text = []
    for i in codes:
        if i == special_id:
            text.append(special_char)
        elif i < len(char_map):
            text.append(char_map[i])
    return "".join(text)


class E2EMetric(nn.Metric):
    """
    Define accuracy metric for end2end network (pgnet).

    Args:
        ignore_space: remove space in prediction and ground truth text if True
        filter_ood: filter out-of-dictionary characters(e.g., '$' for the default digit+en dictionary) in
            ground truth text. Default is True.
        lower: convert GT text to lower case. Recommend to set True if the dictionary does not contains upper letters
        ignore_symbol: Ignore the symbols in the predictions

    Notes:
        Since the OOD characters are skipped during label encoding in data transformation by default,
        filter_ood should be True. (Paddle skipped the OOD character in label encoding and then decoded the label
        indices back to text string, which has no ood character.
    """

    def __init__(
        self,
        character_dict_path: str = None,
        mode: str = "A",
        gt_mat_dir: str = None,
        device_num: int = 1,
        special_id: int = -1,  # 统一替换词表中不存在的字
        special_char: str = "*",  # 统一替换词表中不存在的字
        **kwargs,
    ):
        super().__init__()
        self._evaluator = E2EEvaluator()
        self.clear()

        self.device_num = device_num
        self.all_reduce = AllReduce(reduce="sum") if device_num > 1 else None
        self.metric_names = [
            "recall",
            "precision",
            "f_score",
            "recall_e2e",
            "precision_e2e",
            "f_score_e2e",
            "err",
            "accumulate_recall",
            "accumulate_precision",
            "total_num_gt",
            "total_num_det",
            "total_num_hit",
        ]

        self.mode = mode.lower()
        assert self.mode in ["a", "b"], f"invalid mode: {mode}"
        if self.mode == "B":
            assert osp.exists(gt_mat_dir)
        self.gt_mat_dir = gt_mat_dir

        # TODO: use parsed dictionary object
        if character_dict_path is None:
            self.char_list = "0123456789abcdefghijklmnopqrstuvwxyz"
        else:
            self.char_list = [line.rstrip("\n\r") for line in open(character_dict_path, "r")]
        self.special_id = special_id
        self.special_char = special_char
        assert special_id < 0 or special_id >= len(self.char_list)
        assert special_char not in self.char_list
        # self.code2text = lambda code: ''.join(self.char_list[i] for i in code if i < len(self.char_list))
        self.code2text = partial(code2text, char_map=self.char_list, special_id=special_id, special_char=special_char)

    def clear(self):
        self.results = []

    def update(self, pred: tuple, gt: tuple):
        if self.mode == "a":
            new_gt = []
            for item in gt[1:]:  # gt[0]: img_ids
                if isinstance(item, Tensor):
                    item = item.asnumpy()
                new_gt.append(item)

            for gt_polys, gt_texts, gt_ignore_tags, det_polys, det_texts in zip(*new_gt, pred["polys"], pred["texts"]):
                gts, dets = [], []

                for poly, code, ignore in zip(gt_polys, gt_texts, gt_ignore_tags):
                    text = code if isinstance(code, str) else self.code2text(code)
                    gts.append((Polygon(poly.astype(np.int32)), text, ignore))

                for poly, text in zip(det_polys, det_texts):
                    dets.append((Polygon(poly.astype(np.int32)), text))

                self.results.append((gts, dets))

        else:
            for img_id, det_polys, det_texts in zip(gt[0].asnumpy(), pred["polys"], pred["texts"]):
                gts, dets = [], []

                gt_polys, gt_texts, gt_ignore_tags = read_mat(osp.join(self.gt_mat_dir, f"poly_gt_img{img_id}.mat"))
                for poly, text, ignore in zip(gt_polys, gt_texts, gt_ignore_tags):
                    gts.append((poly, text.lower(), ignore))

                for poly, text in zip(det_polys, det_texts):
                    dets.append((Polygon(poly.astype(np.int32)), text))

                self.results.append((gts, dets))

    def eval(self) -> dict:
        total_recall, total_precision, total_num_gt, total_num_det, total_num_hit = self._evaluator(self.results)

        if self.all_reduce:
            total_recall    = int(self.all_reduce(Tensor(total_recall   , dtype=dtype.float32)).asnumpy())
            total_precision = int(self.all_reduce(Tensor(total_precision, dtype=dtype.float32)).asnumpy())
            total_num_gt    = int(self.all_reduce(Tensor(total_num_gt   , dtype=dtype.float32)).asnumpy())
            total_num_det   = int(self.all_reduce(Tensor(total_num_det  , dtype=dtype.float32)).asnumpy())
            total_num_hit   = int(self.all_reduce(Tensor(total_num_hit  , dtype=dtype.float32)).asnumpy())

        seqerr = 1 - _safe_divide(total_num_hit, total_recall)

        recall    = _safe_divide(total_recall, total_num_gt)
        precision = _safe_divide(total_precision, total_num_det)
        f_score   = _safe_divide(2 * precision * recall, precision + recall)

        recall_e2e    = _safe_divide(total_num_hit, total_num_gt)
        precision_e2e = _safe_divide(total_num_hit, total_num_det)
        f_score_e2e   = _safe_divide(2 * precision_e2e * recall_e2e, precision_e2e + recall_e2e)

        final = {
            'accumulate_recall': total_recall,
            'accumulate_precision': total_precision,

            'total_num_gt' : total_num_gt,
            'total_num_det': total_num_det,
            'total_num_hit': total_num_hit,

            'err': seqerr,

            'recall'   : recall,
            'precision': precision,
            'f_score'  : f_score,

            'recall_e2e'   : recall_e2e,
            'precision_e2e': precision_e2e,
            'f_score_e2e'  : f_score_e2e
        }
        return final
