from itertools import groupby
from typing import List, Tuple

import cv2
import numpy as np
from skimage.morphology._skeletonize import thin

from mindocr.structures.polygon import expand_poly_along_width, point_pair2poly

from .share import sort_and_expand_with_direction_v2, sort_with_direction


def instance_ctc_greedy_decoder(
    tcl: np.ndarray, score_map: np.ndarray, keep_blank_in_tcl=True
) -> Tuple[np.ndarray, float, List[int]]:
    """用CTC贪心策略从一个text center line中解码字符"""
    # 相比fast，少了tcl_align

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

    # 缩减tcl长度，不同于fast
    cur = 0
    keep = []
    for k, v_ in groupby(labels):
        current_len = len(list(v_))
        if keep_blank_in_tcl or k != pad_value:  # 是否保留blank处的点
            keep.append(cur + current_len // 2)
        cur += current_len
    keep_tcl = tcl[keep]

    return keep_tcl, keep_score, keep_text_code


def ctc_decoder_for_image(
    all_tcl_points: List[np.ndarray], score_map: np.ndarray, keep_blank_in_tcl: bool = True
) -> Tuple[List[np.ndarray], List[List[int]]]:
    """用CTC贪心策略从每个text center line中解码字符"""
    texts_tcl, texts_score, texts_code = [], [], []
    for tcl in all_tcl_points:
        keep_tcl, keep_score, keep_text_code = instance_ctc_greedy_decoder(
            tcl, score_map, keep_blank_in_tcl=keep_blank_in_tcl
        )

        texts_tcl.append(keep_tcl)
        texts_score.append(keep_score)
        texts_code.append(keep_text_code)

    return texts_tcl, texts_score, texts_code


def curved_rule(tcl_points: np.ndarray, f_direction: np.ndarray, f_tcl_map: np.ndarray, is_expand=True) -> np.ndarray:
    if is_expand:
        tcl_points_sorted = sort_and_expand_with_direction_v2(tcl_points, f_direction, f_tcl_map)
    else:
        tcl_points_sorted, _ = sort_with_direction(tcl_points, f_direction)
    return tcl_points_sorted


def horizontal_rule(tcl_points: np.ndarray, f_direction: np.ndarray) -> np.ndarray:
    ## add rule here
    # main_direction = extract_main_direction(tcl_points, f_direction)  # y x
    # reference_directin = np.array([0, 1]).reshape([-1, 2])  # y x
    # is_h_angle = abs(np.sum(main_direction * reference_directin)) < math.cos(math.pi / 180 * 70)

    max_y, max_x = np.max(tcl_points, axis=0)
    min_y, min_x = np.min(tcl_points, axis=0)
    is_h_len = (max_y - min_y) < (1.5 * (max_x - min_x))

    if is_h_len:
        xs = np.unique(tcl_points[:, 1])      # [n]
        tmp = tcl_points[:, [1]] == xs[None]  # [m, n]
        ys = tmp.mean(0).astype(xs.dtype)     # [n]
    else:
        ys = np.unique(tcl_points[:, 0])      # [n]
        tmp = tcl_points[:, [0]] == ys[None]  # [m, n]
        xs = tmp.mean(0).astype(ys.dtype)     # [n]
    tcl_points_final = np.stack([ys, xs], axis=-1)  # [n, 2]

    tcl_points_sorted, _ = sort_with_direction(tcl_points_final, f_direction)
    return tcl_points_sorted


def generate_pivot_list_slow(
    f_score: np.ndarray,
    f_char_maps: np.ndarray,
    f_direction: np.ndarray,
    score_thresh: float = 0.5,
    is_curved: bool = True,
) -> Tuple[List[np.ndarray], List[float], List[List[int]]]:
    """用CTC贪心策略解码预测

    Args:
        f_score (np.ndarray): [1, h, w]
        f_char_maps (np.ndarray): [c, h, w]
        f_direction (np.ndarray): [2, h, w]
        score_thresh (float, optional): _description_. Defaults to 0.5.
        is_curved (bool, optional): _description_. Defaults to True.

    Returns:
        Tuple[List[np.ndarray], List[int]]: _description_
    """
    f_tcl_map = f_score[0] > score_thresh
    skeleton_map = thin(f_tcl_map).astype(np.uint8)  # 瘦成n条线
    count, tcl_id_map = cv2.connectedComponents(skeleton_map, connectivity=8)

    # get TCL Instance
    all_tcl_points = []
    if count > 0:
        f_direction = f_direction.transpose(1, 2, 0)
        for id in range(1, count):
            tcl_points = np.stack(np.where(tcl_id_map == id), axis=-1)
            if len(tcl_points) < 3:
                continue

            if is_curved:
                all_tcl_points.append(curved_rule(tcl_points, f_direction, f_tcl_map, is_expand=True))
            else:
                all_tcl_points.append(horizontal_rule(tcl_points, f_direction))

    # use decoder to filter backgroud points.
    texts_tcl, texts_score, texts_code = ctc_decoder_for_image(all_tcl_points, f_char_maps.transpose(1, 2, 0))

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

    f_border = f_border.transpose(1, 2, 0)

    keep_polys, keep_scores, keep_texts = [], [], []
    for center_line, score, text in zip(texts_tcl, texts_score, texts):
        if len(center_line) == 1:  # FIXME why？
            center_line = np.concatenate([center_line, center_line])

        if len(text) < 2:
            continue

        keep_scores.append(score)
        keep_texts.append(text)

        point_pair_list = []
        for point in center_line:
            offset = f_border[point[0], point[1]].reshape(2, 2)

            if dataset == "totaltext":  # FIXME why？
                offset_length = np.linalg.norm(offset, axis=1, keepdims=True)
                expand_length = np.clip(offset_length * 0.2, 0.5, 3.0)
                offset *= 1 + expand_length / offset_length

            point_pair = (point + offset)[:, ::-1] * ratio_wh
            point_pair_list.append(point_pair)

        poly = point_pair2poly(point_pair_list)
        poly = expand_poly_along_width(poly, shrink_ratio_of_width=0.2)
        poly[:, 0] = np.clip(poly[:, 0], 0, src_w)
        poly[:, 1] = np.clip(poly[:, 1], 0, src_h)
        poly = np.round(poly).astype(np.int32)

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
