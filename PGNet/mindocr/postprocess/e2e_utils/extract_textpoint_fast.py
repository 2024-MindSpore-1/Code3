from itertools import groupby
from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology._skeletonize import thin

from mindocr.structures.polygon import expand_poly_along_width, point_pair2poly
from .share import sort_and_expand_with_direction_v2


def tcl_align(tcl: np.ndarray) -> np.ndarray:
    """沿 中心线间断点 填充中心线，使每两点间跨度不超过1像素"""
    insert_num = 0
    for idx in range(len(tcl) - 1):
        tmp_idx = idx + insert_num

        stride = tcl[tmp_idx] - tcl[tmp_idx + 1]  # 沿间断点的向量
        max_points = int(np.max(np.abs(stride)))  # 间隔长度

        if max_points > 1:
            # 插入向量经过的点
            stride = stride / max_points
            insert_value = tcl[tmp_idx] - np.arange(1, max_points)[:, None] * stride[None]
            tcl = np.concatenate([tcl[: tmp_idx + 1], insert_value.astype(tcl.dtype), tcl[tmp_idx + 1 :]])
        insert_num += max_points - 1
    return tcl


def instance_ctc_greedy_decoder(
    tcl: np.ndarray, score_map: np.ndarray, pts_num: int = 6, point_gather_mode: str = None
) -> Tuple[np.ndarray, float, List[int]]:
    """用CTC贪心策略从一个text center line中解码字符"""
    if point_gather_mode == "align":
        tcl = tcl_align(tcl)

    scores = score_map[tcl[:, 0], tcl[:, 1]]
    labels = scores.argmax(1)

    pad_value = score_map.shape[-1] - 1  # 37 -> 36
    keep_scores, keep_text_code = [], []
    cur = 0
    for k, v_ in groupby(labels):
        num_v = len(list(v_))
        if k != pad_value:
            keep_scores.append(scores[cur : cur + num_v, k])
            keep_text_code.append(k)
        cur += num_v
    if len(keep_scores) > 0:
        keep_score = float(np.mean(np.concatenate(keep_scores)))
    else:
        keep_score = 0.0

    # 缩减tcl长度
    detal = len(tcl) // (pts_num - 1)
    keep = [detal * i for i in range(pts_num - 1)] + [-1]
    keep_tcl = tcl[keep]

    return keep_tcl, keep_score, keep_text_code


def ctc_decoder_for_image(
    all_tcl_points: List[np.ndarray], score_map: np.ndarray, pts_num: int = 6, point_gather_mode: str = None
) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
    """用CTC贪心策略从每个text center line中解码字符"""
    texts_tcl, texts_score, texts_code = [], [], []
    for tcl in all_tcl_points:
        if len(tcl) < pts_num:
            continue

        keep_tcl, keep_score, keep_text_code = instance_ctc_greedy_decoder(
            tcl, score_map, pts_num=pts_num, point_gather_mode=point_gather_mode
        )

        if len(keep_text_code) < 2:  # FIXME 可以不加这个限制
            continue

        texts_tcl.append(keep_tcl)
        texts_score.append(keep_score)
        texts_code.append(keep_text_code)

    return texts_tcl, texts_score, texts_code


def generate_pivot_list_fast(
    f_score: np.ndarray,
    f_char_maps: np.ndarray,
    f_direction: np.ndarray,
    score_thresh: float = 0.5,
    point_gather_mode: str = None,
) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
    """用CTC贪心策略解码预测

    Args:
        f_score (np.ndarray): [1, h, w]
        f_char_maps (np.ndarray): [c, h, w]
        f_direction (np.ndarray): [2, h, w]
        score_thresh (float, optional): _description_. Defaults to 0.5.
        point_gather_mode (str, optional): _description_. Defaults to None.

    Returns:
        Tuple[List[np.ndarray], List[int]]: _description_
    """
    f_tcl_map = f_score[0] > score_thresh
    skeleton_map = thin(f_tcl_map).astype(np.uint8)  # 瘦成n条线
    count, tcl_id_map = cv2.connectedComponents(skeleton_map, connectivity=8)  # NOTE count是包含背景的

    # get TCL Instance
    all_tcl_points = []
    if count > 0:
        f_direction = f_direction.transpose(1, 2, 0)
        for id in range(1, count):
            tcl_points = np.stack(np.where(tcl_id_map == id), axis=-1)
            if len(tcl_points) < 3:
                continue
            all_tcl_points.append(sort_and_expand_with_direction_v2(tcl_points, f_direction, f_tcl_map))

    texts_tcl, texts_score, texts_code = ctc_decoder_for_image(
        all_tcl_points, f_char_maps.transpose(1, 2, 0), point_gather_mode=point_gather_mode
    )
    return texts_tcl, texts_score, texts_code


def restore_poly(
    texts_tcl: List[np.ndarray],
    texts_score: List[float],
    texts: List[str],
    f_border: np.ndarray,
    shape_list: np.ndarray,
    dataset: str,
) -> Tuple[list]:
    """将文本中线扩展成polygon

    Args:
        texts_tcl (List[np.ndarray]): text center line
        texts_score (List[float]): _description_
        texts (List[str]): _description_
        f_border (np.ndarray): [4, h, w]
        shape_list (np.ndarray): [src_h, src_w, scale_h, sclae_w]
        dataset (str): _description_

    Raises:
        NotImplementedError: 方法目前仅针对dataset为['partvgg', 'totaltext']的情况

    Returns:
        Tuple[list]: _description_
    """
    src_h, src_w, ratio_h, ratio_w = shape_list
    ratio_wh = 4 / np.array([[ratio_w, ratio_h]])

    offset_expand = 1.2 if dataset == "totaltext" else 1.0
    f_border = f_border.transpose(1, 2, 0) * offset_expand

    keep_polys, keep_scores, keep_texts = [], [], []
    for center_line, score, text in zip(texts_tcl, texts_score, texts):
        if len(text) < 2:
            continue

        keep_scores.append(score)
        keep_texts.append(text)

        point_pair_list = []
        for point in center_line:
            offset = f_border[point[0], point[1]].reshape(2, 2)
            point_pair = (point + offset)[:, ::-1] * ratio_wh  # yx -> xy
            point_pair_list.append(point_pair)

        poly = point_pair2poly(point_pair_list)
        poly = expand_poly_along_width(poly, shrink_ratio_of_width=0.2)
        poly[:, 0] = np.clip(poly[:, 0], 0, src_w)
        poly[:, 1] = np.clip(poly[:, 1], 0, src_h)
        # poly = np.round(poly).astype(np.int32)

        if dataset == "partvgg":
            middle_point = len(poly) // 2
            poly = poly[[0, middle_point - 1, middle_point, -1]]
            keep_polys.append(poly)
        elif dataset == "totaltext":
            keep_polys.append(poly)
        else:
            print("--> Not supported format.")
            raise NotImplementedError

    return keep_polys, keep_scores, keep_texts
