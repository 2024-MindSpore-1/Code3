import json
import math
from typing import List

import cv2
import numpy as np
from skimage.morphology._skeletonize import thin

from mindocr.postprocess.e2e_utils.extract_textpoint_fast import sort_and_expand_with_direction_v2, tcl_align
from mindocr.structures.polygon import (
    gen_min_area_quad_from_poly,
    get_quads_height,
    get_quads_width,
    poly2quads,
    shrink_poly_by_quad,
)

from .det_transforms import DetLabelEncode

__all__ = ["PGLabelEncode", "PGImageAUG", "PGProcessTrain"]


def quad_area(quad: np.ndarray) -> float:
    """compute area of a polygon"""
    edge = [
        (quad[1, 0] - quad[0, 0]) * (quad[1, 1] + quad[0, 1]),
        (quad[2, 0] - quad[1, 0]) * (quad[2, 1] + quad[1, 1]),
        (quad[3, 0] - quad[2, 0]) * (quad[3, 1] + quad[2, 1]),
        (quad[0, 0] - quad[3, 0]) * (quad[0, 1] + quad[3, 1]),
    ]
    return np.sum(edge) / 2.0


def vector_angle(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """计算向量夹角"""
    vector = point2 - point1
    return np.arctan2(vector[1], vector[0])


def line_cross_point_with_angle(angle: np.ndarray, point: np.ndarray) -> list:
    """根据 点 和 斜率 计算直线的一般式: ax + by + c = 0"""
    sin, cos = np.sin(angle), np.cos(angle)
    return [sin, -cos, cos * point[..., 1] - sin * point[..., 0]]


def line_cross_two_point(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """计算过两点的直线: ax + by + c = 0."""
    angle = vector_angle(point1, point2)
    line = line_cross_point_with_angle(angle, point1)
    return line


def average_vertical_angle(quad: np.ndarray) -> np.ndarray:
    """一个四边形左右两边的平均角度（不一定是直观上的左右，而是文字方向的左右）"""
    p0, p1, p2, p3 = quad
    angle_left  = vector_angle(p3, p0)
    angle_right = vector_angle(p2, p1)
    return (angle_left + angle_right) / 2


def cross_point_of_two_lines(line1: List[float], line2: List[float]) -> np.ndarray:
    """计算两直线交点"""
    a1, b1, c1 = line1
    a2, b2, c2 = line2
    d = a1 * b2 - a2 * b1
    x = (b1 * c2 - b2 * c1) / d
    y = (a2 * c1 - a1 * c2) / d
    return np.array([x, y])


class PGLabelEncode(DetLabelEncode):
    """解码标注，并统一一张图片中所有标注的polygon坐标点数量
    need key:
        label,
        image

    add key:
        polys,
        texts,
        ignore_tags
    """

    def expand_points_num(self, boxes: list) -> list:
        if len(boxes) == 0:
            return boxes

        max_points_num = max(len(b) for b in boxes)
        if max_points_num % 2 != 0:
            max_points_num += 1

        ex_boxes = []
        for b in boxes:
            num_pad = max_points_num - len(b)
            if num_pad > 0:
                # NOTE 只要不涉及平均字符间距，就没问题
                # 在序列的首尾进行拷贝填充，相当于在左边填充面积为0的四边形
                left_pad  = num_pad // 2
                right_pad = num_pad - left_pad
                ex_box = [b[0]] * left_pad + b + [b[-1]] * right_pad
            else:
                ex_box = b
            ex_boxes.append(ex_box)
        return ex_boxes

    def __call__(self, data: dict) -> dict:
        h, w = data["image"].shape[:2]
        label = json.loads(data["label"])

        boxes, texts, ignore_tags = [], [], []
        for info in label:
            box = np.array(info["points"], dtype=np.float32)
            text = info["transcription"]  # NOTE ICDAR2019有字符为空的，你敢信？！
            ignore = text in ["", "*", "###"] or len(box) % 2 != 0  # NOTE 排除点个数是奇数的polygon，因为要拆分成若干四边形
            # if len(box) % 2 != 0:  # NOTE 排除点个数是奇数的polygon，因为要拆分成若干四边形
            #     return None  # TODO
            if np.any(box[:, 0] > w + 10) or np.any(box[:, 1] > h + 10):
                return None  # NOTE 有的图片旋转了90度，但标注没变，比如icdar2019_train的41/1491/5194

            box[:, 0] = np.clip(box[:, 0], 0, w - 1)
            box[:, 1] = np.clip(box[:, 1], 0, h - 1)
            if abs(quad_area(gen_min_area_quad_from_poly(box))) < 1:
                print("invalid poly")
                continue

            boxes.append(box.tolist())
            texts.append(text)
            ignore_tags.append(ignore)

        if len(boxes) == 0 or all(ignore_tags):
            return None

        data["polys"] = np.array(self.expand_points_num(boxes), dtype=np.float32)
        data["texts"] = texts
        data["ignore_tags"] = np.array(ignore_tags, dtype=bool)
        return data


class PGImageAUG:
    """resize & crop & pad

    need key:
        image,
        polys,
        ignore_tags,
        texts

    modify key:
        image,
        polys,
        ignore_tags,
        texts
    """

    def __init__(
        self,
        input_size: int = 512,
        use_resize: bool = True,
        use_crop: bool = False,  # 尽量不要用
        min_crop_size: int = 24,
        debug: bool = False,
        **kwargs
    ):
        self.debug = debug
        self.input_size = input_size
        self.min_edge_len = input_size * 0.5
        self.use_resize = use_resize
        self.use_crop = use_crop
        if use_crop:
            self.min_crop_size = min_crop_size
        self.asp_scales = np.arange(1.0, 1.55, 0.1)
        self.rand_scales = np.array([0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.mean = np.array([0.485 * 255, 0.456 * 255, 0.406 * 255], dtype=np.float32)  # RGB
        self.std  = np.array([0.229 * 255, 0.224 * 255, 0.225 * 255], dtype=np.float32)  # RGB

    def crop_area(self, image, polys, ignores, texts, max_tries=25):
        """这个实现不太好
        make random crop from the input image
        :param image:
        :param polys:  [b,4,2]
        :param ignores:
        :param crop_background:
        :param max_tries: 50 -> 25
        :return:
        """
        h, w, _ = image.shape

        # 分坐标轴考虑，简化了crop的情况
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)

        # mask各poly的外接矩形（平行于坐标轴的）
        tmp_polys = np.round(polys).astype(np.int32)
        for poly in tmp_polys:
            w_array[np.min(poly[:, 0]) : np.max(poly[:, 0])] = 1
            h_array[np.min(poly[:, 1]) : np.max(poly[:, 1])] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return image, polys, ignores, texts

        for _ in range(max_tries):
            xx = np.random.choice(w_axis, size=2)
            xmin = np.min(xx)
            xmax = np.max(xx)

            yy = np.random.choice(h_axis, size=2)
            ymin = np.min(yy)
            ymax = np.max(yy)

            if (xmax - xmin < self.min_crop_size) or (ymax - ymin < self.min_crop_size):
                continue

            keep = []
            if polys.shape[0] > 0:
                # 筛选每个点都在crop范围内的poly
                poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                  & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
                keep = np.where(np.all(poly_axis_in_area, axis=1))[0]

            if len(keep) == 0:
                continue  # no text in this area

            image   = image[ymin:ymax + 1, xmin:xmax + 1]
            texts   = [texts[i] for i in keep]
            ignores = ignores[keep]
            polys   = polys[keep] - np.array([xmin, ymin])
            break

        return image, polys, ignores, texts

    def __call__(self, data):
        image = data["image"]
        polys = data["polys"]
        ignores = data["ignore_tags"]
        texts = data["texts"]

        # set aspect ratio and keep area fix
        if self.debug:
            asp_scale = 1.2
        else:
            asp_scale = np.random.choice(self.asp_scales)
            if np.random.rand() < 0.5:
                asp_scale = 1.0 / asp_scale
        asp_scale = math.sqrt(asp_scale)

        asp_x, asp_y = asp_scale, 1.0 / asp_scale
        image = cv2.resize(image, dsize=None, fx=asp_x, fy=asp_y)
        polys *= np.array([asp_x, asp_y])

        max_hw = min(image.shape[:2])
        if self.use_resize:
            if max_hw < 200:
                ratio = 200 / max_hw
                image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
                polys *= ratio
            elif max_hw > 512:
                ratio = 512 / max_hw
                image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
                polys *= ratio
        elif self.use_crop:
            if max_hw > 2048:
                ratio = 2048 / max_hw
                image = cv2.resize(image, dsize=None, fx=ratio, fy=ratio)
                polys *= ratio
            if min(image.shape[:2]) < 16:
                return None
            image, polys, ignores, texts = self.crop_area(image, polys, ignores, texts)
            if polys.shape[0] == 0 or np.all(ignores):
                return None

        # resize image
        rz_scale = self.input_size / max(image.shape[:2])
        if not self.debug:
            rz_scale *= np.random.choice(self.rand_scales)
        if min(image.shape[:2]) * rz_scale < self.input_size * 0.5:
            return None
        image = cv2.resize(image, dsize=None, fx=rz_scale, fy=rz_scale)
        polys *= rz_scale

        # gaussian blur
        if not self.debug and np.random.rand() < 0.05:
            ks = np.random.permutation(5)[0] + 1
            ks = int(ks / 2) * 2 + 1
            image = cv2.GaussianBlur(image, ksize=(ks, ks), sigmaX=0, sigmaY=0)
        # brightness  # TODO
        if not self.debug and np.random.rand() < 0.05:
            image = image * (0.5 + np.random.rand())
            image = np.clip(image, 0.0, 255.0)

        image = (image - self.mean) / self.std  # rgb

        # Random the start position
        new_h, new_w, _ = image.shape
        sh, sw = 0, 0
        if not self.debug:
            sh = int(np.random.rand() * (self.input_size - new_h))
            sw = int(np.random.rand() * (self.input_size - new_w))

        # Padding the image to [input_size, input_size]
        im_padded = np.zeros((self.input_size, self.input_size, 3), dtype=np.float32)
        im_padded[sh : sh + new_h, sw : sw + new_w] = image.copy()
        polys += np.array([sw, sh])

        data["image"] = im_padded
        data["polys"] = polys
        data["ignore_tags"] = ignores
        data["texts"] = texts
        return data


class PGProcessTrain:
    """产生各种ground truth
    Args:
        character_dict_path (str): 词表路径
        ds_ratio (float, optional): 模型输出相对于输入的缩放系数. Defaults to 0.25.
        min_quad_size (int, optional): polygon外接旋转矩形最小边长，太小就忽略. Defaults to 4.
        max_quad_size (int, optional): polygon外接旋转矩形最大边长，太大也忽略. Defaults to 512.
        tcl_ratio_h (float, optional): tcl相对于polygon在高度上的缩放系数. Defaults to 0.3.
        tcl_ratio_w (float, optional): tcl相对于polygon在宽度上的缩放系数. Defaults to 0.15.
        max_text_len (int, optional): 单个文本字符数上限. Defaults to 30.
        max_text_nums (int, optional): 所有文本数量上限. Defaults to 50.
        tcl_len (int, optional): 单个文本ctc_tcl的长度上限. Defaults to 64.

    need key:
        image,
        polys,
        texts,
        ignore_tags

    add key:
        training_mask,
        tcl_map,
        tbo_map,
        tdo_map,
        ctc_points,
        ctc_masks,
        ctc_labels
    """

    def __init__(
        self,
        character_dict_path: str,
        ds_ratio: float = 0.25,
        min_quad_size: int = 4,
        max_quad_size: int = 512,
        tcl_ratio_h: float = 0.3,
        tcl_ratio_w: float = 0.15,
        max_text_len: int = 30,
        max_text_nums: int = 50,
        tcl_len: int = 64,
        point_gather_mode: str = "align",
        debug: bool = False,
        **kwargs
    ):
        self.debug = debug
        self.ds_ratio = ds_ratio
        self.min_quad_size = min_quad_size * ds_ratio
        self.max_quad_size = max_quad_size * ds_ratio
        self.tcl_ratio_h = tcl_ratio_h
        self.tcl_ratio_w = tcl_ratio_w

        self.char_dict = {line.rstrip("\n\r"): i for i, line in enumerate(open(character_dict_path, "r"))}
        self.pad_num = len(self.char_dict)

        self.max_text_len = max_text_len
        self.max_text_nums = max_text_nums
        self.tcl_len = tcl_len
        self.point_gather_mode = point_gather_mode

    # NOTE 和 det_transforms.py 里的 BorderMap 不同
    @staticmethod
    def gen_quad_tbo(quad: np.ndarray, tcl_quad_mask: np.ndarray, tbo_map: np.ndarray):
        """子文本框偏移text border offset (TBO)"""
        ys, xs = np.nonzero(tcl_quad_mask)
        xy_in_tcl = np.stack([xs, ys], axis=-1)

        angle = average_vertical_angle(quad)  # average angle of left and right line.
        lines = line_cross_point_with_angle(angle, xy_in_tcl)
        upper_line = line_cross_two_point(quad[0], quad[1])
        lower_line = line_cross_two_point(quad[3], quad[2])

        cross_point_upper = cross_point_of_two_lines(upper_line, lines)  # 两直线交点
        cross_point_lower = cross_point_of_two_lines(lower_line, lines)

        upper_offset_x, upper_offset_y = cross_point_upper - xy_in_tcl.T
        lower_offset_x, lower_offset_y = cross_point_lower - xy_in_tcl.T

        quad_h = get_quads_height(quad[None])[0]
        quad_w = get_quads_width (quad[None])[0]
        offset_norm = 2 / max(min(quad_h, quad_w), 1)

        tbo_map[ys, xs, 0] = upper_offset_y
        tbo_map[ys, xs, 1] = upper_offset_x
        tbo_map[ys, xs, 2] = lower_offset_y
        tbo_map[ys, xs, 3] = lower_offset_x
        tbo_map[ys, xs, 4] = offset_norm
        # if np.isinf(tbo_map).any() or np.isnan(tbo_map).any():
        #     a = 0  # 对于一些非严格上下配对的polygon，会产生离谱的数据，例如icdar2019-train-2160

    def gather_ctc_points_v2(self, poly: np.ndarray, ds_h: int, ds_w: int, avg_height: float) -> np.ndarray:
        """Find the center point of poly as key_points, then fit and gather."""
        poly = poly[..., ::-1]  # xy -> yx
        ctc_points = [0.5 * (poly[idx] + poly[-1 - idx]) for idx in range(poly.shape[0] // 2)]
        ctc_points = tcl_align(ctc_points)  # 使ctc_points连续

        if not self.debug and np.random.rand() < 0.2 and avg_height >= 3:
            # 加入上下抖动
            noise_h = (np.random.rand(len(ctc_points)) - 0.5) * avg_height * 0.3
            ctc_points[:, 0] = np.clip(np.round(ctc_points[:, 0] + noise_h), 0, ds_h - 1)

        ctc_points = np.round(ctc_points).astype(int)
        return ctc_points

    def gather_ctc_points_v3(
        self, tdo_map: np.ndarray, tcl_poly: np.ndarray, ds_h: int, ds_w: int, avg_height: float
    ) -> np.ndarray:
        """Find the center point of poly as key_points, then fit and gather."""
        tcl_map = np.zeros(
            (int(ds_h / self.ds_ratio), int(ds_w / self.ds_ratio)), dtype=np.float32
        )  # 防止图片缩的太小导致polygon消失
        cv2.fillPoly(tcl_map, [np.round(tcl_poly / self.ds_ratio).astype(np.int32)], 1.0)
        tcl_map = cv2.resize(tcl_map, dsize=None, fx=self.ds_ratio, fy=self.ds_ratio) > 1e-3

        skeleton_map = thin(tcl_map.astype(np.uint8)).astype(np.uint8)  # 瘦成一条线
        ctc_points = np.stack(np.nonzero(skeleton_map), axis=-1)  # NOTE 无序 [n, 2] yx
        if len(ctc_points) < 3:
            return None
        ctc_points = sort_and_expand_with_direction_v2(ctc_points, tdo_map, tcl_map).astype(np.float32)  # 给ctc_points排序
        ctc_points = tcl_align(ctc_points)  # 使ctc_points连续

        # 加入上下左右抖动
        if not self.debug:
            half_num_poly = tcl_poly.shape[0] // 2
            avg_width = (
                (
                    np.abs(tcl_poly[0, 0] - tcl_poly[half_num_poly - 1, 0])
                    + np.abs(tcl_poly[-1, 0] - tcl_poly[half_num_poly, 0])
                )
                // 2
                * 0.2
            )
            noise_h = (np.random.rand(len(ctc_points)) - 0.5) * avg_height  # FIXME 这怎么和V2还不一样？
            noise_w = (np.random.rand(1) - 0.5) * avg_width
            ctc_points[:, 0] = np.clip(ctc_points[:, 0] + noise_h, 0, ds_h - 1)
            ctc_points[:, 1] = np.clip(ctc_points[:, 1] + noise_w, 0, ds_w - 1)

        ctc_points = np.round(ctc_points).astype(int)
        return ctc_points

    def __call__(self, data: dict) -> dict:
        h, w = data["image"].shape[:2]
        tcl_map = np.zeros((h, w), dtype=np.float32)  # NOTE 先在未缩放的map上画poly

        ds_h, ds_w = int(h * self.ds_ratio), int(w * self.ds_ratio)
        training_mask = np.ones((ds_h, ds_w), dtype=np.float32)
        tbo_map = np.zeros((ds_h, ds_w, 5), dtype=np.float32)
        tdo_map = np.zeros((ds_h, ds_w, 3), dtype=np.float32)  # direction & norm
        tdo_map[..., -1] = 1.0
        ctc_points, ctc_masks, ctc_labels = [], [], []

        for poly, ignore, text in zip(data["polys"], data["ignore_tags"], data["texts"]):
            assert len(poly) % 2 == 0
            poly *= self.ds_ratio
            min_area_quad = gen_min_area_quad_from_poly(poly)  # 最小外接旋转矩形
            min_area_quad_h = get_quads_height(min_area_quad[None])[0]
            min_area_quad_w = get_quads_width (min_area_quad[None])[0]
            min_area_wh = min(min_area_quad_h, min_area_quad_w)

            if min_area_wh < self.min_quad_size or min_area_wh > self.max_quad_size:
                continue

            if ignore:
                # NOTE ploy 得有个num维度，即 P,2 -> 1,P,2 (对于total_text数据集，P=14)
                cv2.fillPoly(training_mask, [poly.astype(np.int32)], 0.15)  # NOTE smooth
                continue

            ## 文本中线区域 - text center line（TCL）
            tcl_poly, quad_index = shrink_poly_by_quad(poly, self.tcl_ratio_h, self.tcl_ratio_w)
            cv2.fillPoly(tcl_map, [np.round(tcl_poly / self.ds_ratio).astype(np.int32)], 1.0)

            ## 文本边框偏移 - text border offset（TBO）
            tcl_quads  = poly2quads(tcl_poly)
            poly_quads = poly2quads(poly)
            for tcl_q, q_id in zip(np.round(tcl_quads).astype(np.int32), quad_index):
                quad_mask = np.zeros((ds_h, ds_w), dtype=np.float32)
                cv2.fillPoly(quad_mask, [tcl_q], 1.0)
                self.gen_quad_tbo(poly_quads[q_id], quad_mask, tbo_map)

            ## 文本方向偏移 - text direction offset（TDO）
            widths, heights = get_quads_width(poly_quads), get_quads_height(poly_quads)
            norm_width  = max(1.0, np.sum(widths) / len(text))
            norm_height = min(1.0, 1.0 / np.mean(heights))

            direct_vector = 0.5 * (poly_quads[:, 1] - poly_quads[:, 0] + poly_quads[:, 2] - poly_quads[:, 3])
            direct_vector *= norm_width / (np.linalg.norm(direct_vector, axis=-1, keepdims=True) + 1e-6)

            for quad, direct in zip(np.round(poly_quads).astype(np.int32), direct_vector.tolist()):
                if not (direct[0] == 0 and direct[1] == 0):  # NOTE 存在pad点
                    direct.append(norm_height)
                    cv2.fillPoly(tdo_map, [quad], direct)

            ## 文本字符分类 - text character classification（TCC）
            tcl_avg_height = max(1.0, np.mean(get_quads_height(tcl_quads)))
            # 得到逐像素连续的中心线
            if self.point_gather_mode == "align":
                points = self.gather_ctc_points_v3(tdo_map[:, :, :-1], tcl_poly, ds_h, ds_w, tcl_avg_height)
                if points is None:
                    continue
            else:
                points = self.gather_ctc_points_v2(poly, ds_h, ds_w, tcl_avg_height)
            # 裁剪或填充到指定长度
            num_points = len(points)
            if num_points > self.tcl_len:
                keep = (num_points / self.tcl_len * np.arange(self.tcl_len)).astype(int)
                points = points[keep]
                mask = [True] * self.tcl_len
            else:
                pad_num = self.tcl_len - num_points
                points = np.concatenate([points, np.zeros((pad_num, 2), dtype=points.dtype)])
                mask = [True] * num_points + [False] * pad_num

            # 文本字符转字典索引
            if len(self.char_dict) == 36:
                text = text.lower()  # 若字典长度=36，就取lower
            label = [self.char_dict[c] for c in text if c in self.char_dict]  # 转成字典索引，顺便排除非法字符
            # 裁剪或填充到指定长度
            if len(label) > self.max_text_len:
                label = label[: self.max_text_len]
            else:
                label.extend([self.pad_num] * (self.max_text_len - len(label)))

            ctc_points.append(points)
            ctc_masks.append(mask)
            ctc_labels.append(label)

        num_valid_text = len(ctc_points)
        if num_valid_text == 0 or num_valid_text > self.max_text_nums:
            return None  # 图片中没有文本，或文本数太多
        if np.isinf(tbo_map).any() or np.isnan(tbo_map).any() or np.isinf(tdo_map).any() or np.isnan(tdo_map).any():
            return None  # 有些标注不是严格上下两两配对的

        # 填充文本标注数量
        num_pad = self.max_text_nums - num_valid_text
        if num_pad > 0:
            ctc_points.extend([np.zeros((self.tcl_len, 2), dtype=int)] * num_pad)
            ctc_masks.extend( [ [   False    ]  *  self.tcl_len      ] * num_pad)
            ctc_labels.extend([ [self.pad_num]  *  self.max_text_len ] * num_pad)

        tcl_map = cv2.resize(tcl_map, dsize=None, fx=self.ds_ratio, fy=self.ds_ratio) > 1e-3  # 再缩放到目标大小

        data["training_mask"] = training_mask[None]  # [1, h, w] float
        data["tcl_map"] = tcl_map[None].astype(np.float32)  # [1, h, w]  float
        data["tbo_map"] = tbo_map.transpose(2, 0, 1)
        data["tdo_map"] = tdo_map.transpose(2, 0, 1)
        data["ctc_points"] = np.array(ctc_points, dtype=np.int64)  # [max_text_num, tcl_len,   2]  int64
        data["ctc_masks" ] = np.array(ctc_masks , dtype=np.bool_)  # [max_text_num, tcl_len     ]  bool
        data["ctc_labels"] = np.array(ctc_labels, dtype=np.int32)  # [max_text_num, max_text_len]  int32
        return data
