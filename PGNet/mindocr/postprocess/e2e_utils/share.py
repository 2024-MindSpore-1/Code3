from typing import Tuple

import numpy as np

# points     : n x 2    , yx-format
# f_direction: h x w x 2, xy-format


def extract_main_direction(points: np.ndarray, f_direction: np.ndarray) -> np.ndarray:
    points_direction = f_direction[points[:, 0], points[:, 1]][:, ::-1]  # x, y -> y, x
    avg_direction = np.mean(points_direction, axis=0, keepdims=True)
    avg_direction /= np.linalg.norm(avg_direction) + 1e-6
    return avg_direction


def sort_part_with_direction(points: np.ndarray, points_direction: np.ndarray) -> Tuple[np.ndarray]:
    avg_direction = np.mean(points_direction, axis=0, keepdims=True)  # [1, 2]
    proj_value = np.sum(points * avg_direction, axis=1)  # [n, 2] -> [n]
    idx = np.argsort(proj_value)
    return points[idx], points_direction[idx]


def sort_with_direction(points: np.ndarray, f_direction: np.ndarray) -> Tuple[np.ndarray]:
    points_direction = f_direction[points[:, 0], points[:, 1]][:, ::-1]  # [n, 2] xy -> yx
    sorted_point, sorted_direction = sort_part_with_direction(points, points_direction)

    half_point_num = len(sorted_point) // 2
    if half_point_num >= 8:  # FIXME 有什么特殊的？
        sorted_fist_part_point, sorted_fist_part_direction = sort_part_with_direction(
            sorted_point[:half_point_num], sorted_direction[:half_point_num])
        sorted_last_part_point, sorted_last_part_direction = sort_part_with_direction(
            sorted_point[half_point_num:], sorted_direction[half_point_num:])
        sorted_point     = np.concatenate([sorted_fist_part_point    , sorted_last_part_point    ])
        sorted_direction = np.concatenate([sorted_fist_part_direction, sorted_last_part_direction])
    return sorted_point, sorted_direction


# 没用到
def sort_and_expand_with_direction(points: np.ndarray, f_direction: np.ndarray) -> np.ndarray:
    """ 按direction排序points，并向两端扩展（因为TCL两端向内收缩了）"""
    sorted_points, points_direction = sort_with_direction(points, f_direction)  # return yx-format

    sub_direction_len = max(len(sorted_points) // 3, 2)
    left_direction = points_direction[ :sub_direction_len, :]
    right_dirction = points_direction[-sub_direction_len:, :]

    left_avg_direction = -np.mean(left_direction, axis=0)
    left_avg_len = np.linalg.norm(left_avg_direction)
    left_start = sorted_points[0]
    left_step = left_avg_direction / (left_avg_len + 1e-6)

    right_avg_direction = np.mean(right_dirction, axis=0)
    right_avg_len = np.linalg.norm(right_avg_direction)
    right_start = sorted_points[-1]
    right_step = right_avg_direction / (right_avg_len + 1e-6)

    append_num = max(int((left_avg_len + right_avg_len) / 2.0 * 0.15), 1)

    h, w, _ = f_direction.shape
    left_list, right_list = [], []
    for i in range(append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).astype(int)
        if ly < h and lx < w and (ly, lx) not in left_list:
            left_list.append((ly, lx))

        ry, rx = np.round(right_start + right_step * (i + 1)).astype(int)
        if ry < h and rx < w and (ry, rx) not in right_list:
            right_list.append((ry, rx))

    all_list = np.concatenate(
        [
            np.array(left_list[::-1], dtype=int).reshape(-1, 2),
            sorted_points,
            np.array(right_list, dtype=int).reshape(-1, 2),
        ],
        dtype=int,
    )
    return all_list


def sort_and_expand_with_direction_v2(points: np.ndarray, f_direction: np.ndarray, tcl_map: np.ndarray) -> np.ndarray:
    """按direction排序points，并向两端扩展（因为TCL两端向内收缩了）
    tcl_map: h x w, bool
    """
    sorted_points, points_direction = sort_with_direction(points, f_direction)  # return yx-format

    sub_direction_len = max(len(sorted_points) // 3, 2)
    left_direction  = points_direction[ :sub_direction_len]
    right_direction = points_direction[-sub_direction_len:]

    left_avg_direction = -np.mean(left_direction, axis=0)
    left_avg_len = np.linalg.norm(left_avg_direction)
    left_start = sorted_points[0]
    left_step = left_avg_direction / (left_avg_len + 1e-6)

    right_avg_direction = np.mean(right_direction, axis=0)
    right_avg_len = np.linalg.norm(right_avg_direction)
    right_start = sorted_points[-1]
    right_step = right_avg_direction / (right_avg_len + 1e-6)

    append_num = max(int((left_avg_len + right_avg_len) * 0.15), 2)

    h, w, _ = f_direction.shape
    left_list = []
    for i in range(append_num):
        ly, lx = np.round(left_start + left_step * (i + 1)).astype(int)
        if ly < h and lx < w and (ly, lx) not in left_list:
            if not tcl_map[ly, lx]:
                break
            left_list.append((ly, lx))

    right_list = []
    for i in range(append_num):
        ry, rx = np.round(right_start + right_step * (i + 1)).astype(int)
        if ry < h and rx < w and (ry, rx) not in right_list:
            if not tcl_map[ry, rx]:
                break
            right_list.append((ry, rx))

    all_list = np.concatenate(
        [
            np.array(left_list[::-1], dtype=int).reshape(-1, 2),
            sorted_points,
            np.array(right_list, dtype=int).reshape(-1, 2),
        ],
        dtype=int,
    )
    return all_list
