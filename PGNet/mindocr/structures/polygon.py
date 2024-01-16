from typing import List, Tuple, Union

import cv2
import numpy as np
from shapely.geometry import Polygon

from mindocr.data.utils.polygon_utils import sort_clockwise

# NOTE 约定
# polygon: 从左上角开始，向右，向下，在向左一圈
# quad: 左上，右上，右下，左下


def area_of_intersection(det_p: Polygon, gt_p: Polygon) -> float:
    p1 = det_p.buffer(0)
    p2 = gt_p.buffer(0)
    return p1.intersection(p2).area


def area_of_union(det_p: Polygon, gt_p: Polygon) -> float:
    p1 = det_p.buffer(0)
    p2 = gt_p.buffer(0)
    return p1.union(p2).area


def iou(det_p: Polygon, gt_p: Polygon) -> float:
    return area_of_intersection(det_p, gt_p) / (1 + area_of_union(det_p, gt_p))


def iod(det_p: Polygon, gt_p: Polygon) -> float:
    """This helper determine the fraction of intersection area over detection area."""
    return area_of_intersection(det_p, gt_p) / (1 + det_p.area)


def get_quads_width(quads: np.ndarray) -> np.ndarray:
    """各四边形的平均上下边长度"""
    widths = 0.5 * (np.linalg.norm(quads[:, 0] - quads[:, 1], axis=1) + 
                    np.linalg.norm(quads[:, 2] - quads[:, 3], axis=1))
    return widths


def get_quads_height(quads: np.ndarray) -> np.ndarray:
    """各四边形的平均高度"""
    heights = 0.5 * (np.linalg.norm(quads[:, 0] - quads[:, 3], axis=1) + 
                     np.linalg.norm(quads[:, 2] - quads[:, 1], axis=1))
    return heights


def gen_min_area_quad_from_poly(poly: Union[np.ndarray, list]) -> np.ndarray:
    """最小外接旋转四边形"""
    if isinstance(poly, list):
        poly = np.array(poly)

    point_num = poly.shape[0]
    if point_num == 4:
        min_area_quad = poly
    else:
        rect = cv2.minAreaRect(poly.astype(np.int32))  # (center (x,y), (width, height), angle of rotation)
        box = np.array(cv2.boxPoints(rect))
        min_area_quad = sort_clockwise(box)
    return min_area_quad


def poly2quads(poly: np.ndarray) -> np.ndarray:
    """组成上下点对，相邻两组构成一个quad（四边形）"""
    point_num = poly.shape[0]

    # 上下点对
    point_pair_list = np.array([[poly[idx], poly[-1 - idx]] for idx in range(point_num // 2)])

    # 相邻两组构成一个quad
    quad_list = []
    for idx in range(point_num // 2 - 1):
        # reshape and adjust to clock-wise
        quad_list.append(point_pair_list[[idx, idx + 1]].reshape(4, 2)[[0, 2, 3, 1]])

    return np.array(quad_list)


def quads2poly(quads: np.ndarray) -> np.ndarray:
    """将若干相连子四边形还原成poly"""
    point_list = [None] * ((1 + len(quads)) * 2)
    point_list[0], point_list[-1] = quads[0, 0], quads[0, 3]
    for idx, quad in enumerate(quads):
        point_list[ 1+idx] = quad[1]
        point_list[-2-idx] = quad[2]
    return np.array(point_list)


def point_pair2poly(point_pair_list: List[np.ndarray]) -> np.ndarray:
    """Transfer vertical point_pairs into poly point in clockwise."""
    point_num = len(point_pair_list) * 2
    point_list = [None] * point_num
    for idx, point_pair in enumerate(point_pair_list):
        point_list[   idx] = point_pair[0]
        point_list[-1-idx] = point_pair[1]
    return np.array(point_list).reshape(-1, 2)


def get_cut_info(upper_edges_len: np.ndarray, cut_len: float) -> Tuple[int, float]:
    """避免遇到expand_poly"""
    for idx, edge_len in enumerate(upper_edges_len):
        if cut_len <= edge_len:
            return idx, cut_len / edge_len
        cut_len -= edge_len
    raise ValueError(f"invalid cut_len: {cut_len}")


def shrink_quad_along_width(
    quad: np.ndarray, begin_width_ratio: float = 0.0, end_width_ratio: float = 1.0
) -> np.ndarray:
    """缩小四边形的 “宽度”"""
    ratio_pair = np.array([[begin_width_ratio], [end_width_ratio]], dtype=np.float32)  # [2， 1]
    upper_edge = quad[0] + (quad[1] - quad[0]) * ratio_pair  # [2, 2]
    lower_edge = quad[3] + (quad[2] - quad[3]) * ratio_pair
    return np.array([upper_edge[0], upper_edge[1], lower_edge[1], lower_edge[0]])  # [4, 2] NOTE 注意顺序


def shrink_poly_by_quad(poly: np.ndarray, ratio_h: float, ratio_w: float) -> Tuple[np.ndarray, List[int]]:
    num_pairs = poly.shape[0] // 2

    ## 先缩放polygon的 “高度”
    ratio_h = np.array([[0.5 - ratio_h / 2], [0.5 + ratio_h / 2]], dtype=np.float32)  # [2, 1]
    tcl_poly = np.zeros_like(poly)  # [p, 2]
    for idx in range(num_pairs):
        point_pair = poly[idx] + (poly[-1-idx] - poly[idx]) * ratio_h
        tcl_poly[   idx] = point_pair[0]
        tcl_poly[-1-idx] = point_pair[1]

    ## 再缩放polygon的 “宽度”
    poly_quads = poly2quads(poly)
    tcl_quads  = poly2quads(tcl_poly)

    upper_edges_len = np.linalg.norm(poly_quads[ :, 0] - poly_quads[ :, 1], axis=-1)  # 子四边形的 上边 的长度，和poly_quads一样
    left_edge_len   = np.linalg.norm(poly_quads[ 0, 0] - poly_quads[ 0, 3])
    right_edge_len  = np.linalg.norm(poly_quads[-1, 1] - poly_quads[-1, 2])

    # FIXME 缩放后的左右边长、未缩放的上边总长，取最小值（why?），再缩放
    shrink_length = min(left_edge_len, right_edge_len, upper_edges_len.sum()) * ratio_w
    left_shrink, right_shrink = shrink_length, sum(upper_edges_len) - shrink_length

    left_idx, left_ratio = get_cut_info(upper_edges_len, left_shrink)
    left_quad = shrink_quad_along_width(tcl_quads[left_idx], left_ratio, 1)

    right_idx, right_ratio = get_cut_info(upper_edges_len, right_shrink)
    right_quad = shrink_quad_along_width(tcl_quads[right_idx], 0, right_ratio)

    if left_idx == right_idx:
        tcl_poly = np.array([left_quad[0], right_quad[1], right_quad[2], left_quad[3]])
    else:
        out_quads = [left_quad] + tcl_quads[left_idx + 1 : right_idx].tolist() + [right_quad]
        tcl_poly = quads2poly(np.array(out_quads))

    quad_index = list(range(left_idx, right_idx + 1))
    return tcl_poly, quad_index


def expand_poly_along_width(poly: np.ndarray, shrink_ratio_of_width: float = 0.3) -> np.ndarray:
    """expand poly along width."""
    left_quad = poly[[0, 1, -2, -1]].copy()
    left_ratio = -shrink_ratio_of_width \
               *  np.linalg.norm(left_quad[0] - left_quad[3]) \
               / (np.linalg.norm(left_quad[0] - left_quad[1]) + 1e-6)
    left_quad_expand = shrink_quad_along_width(left_quad, left_ratio, 1.0)
    poly[ 0] = left_quad_expand[ 0]
    poly[-1] = left_quad_expand[-1]

    half_point_num = poly.shape[0] // 2
    right_quad = poly[half_point_num-2: half_point_num+2].copy()
    right_ratio = 1.0 + shrink_ratio_of_width \
                *  np.linalg.norm(right_quad[0] - right_quad[3]) \
                / (np.linalg.norm(right_quad[0] - right_quad[1]) + 1e-6)
    right_quad_expand = shrink_quad_along_width(right_quad, 0.0, right_ratio)
    poly[half_point_num-1] = right_quad_expand[1]
    poly[half_point_num  ] = right_quad_expand[2]
    return poly
