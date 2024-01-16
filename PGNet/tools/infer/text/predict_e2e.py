"""
End2end text detection & recognition inference

Example:
    $ python tools/infer/text/predict_e2e.py  --image_dir {path_to_img} --e2e_algorithm PG
"""

import json
import logging
import os
from typing import List

import numpy as np
from config import parse_args
from postprocess import Postprocessor
from preprocess import Preprocessor
from shapely.geometry import Polygon
from utils import get_ckpt_file, get_image_paths

import mindspore as ms
from mindspore import ops

from mindocr import build_model
from mindocr.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mindocr.utils.logger import set_logger
from mindocr.utils.visualize import draw_boxes, show_imgs, visualize

# map algorithm name to model name (which can be checked by `mindocr.list_models()`)
# NOTE: Modify it to add new model for inference.
algo_to_model_name = {
    "PG": "pgnet_resnet50",
}
logger = logging.getLogger("mindocr")


class TextEnd2End(object):
    def __init__(self, args):
        # build model for algorithm with pretrained weights or local checkpoint
        ckpt_dir = args.e2e_model_dir
        if ckpt_dir is None:
            pretrained = True
            ckpt_load_path = None
        else:
            ckpt_load_path = get_ckpt_file(ckpt_dir)
            pretrained = False
        assert args.e2e_algorithm in algo_to_model_name, (
            f"Invalid e2e_algorithm {args.e2e_algorithm}. "
            f"Supported algorithms are {list(algo_to_model_name.keys())}"
        )

        model_name = algo_to_model_name[args.e2e_algorithm]
        amp_level = args.e2e_amp_level
        self.model = build_model(model_name, pretrained=pretrained, ckpt_load_path=ckpt_load_path, amp_level=amp_level)
        self.model.set_train(False)

        self.cast_pred_fp32 = amp_level != "O0"
        if self.cast_pred_fp32:
            self.cast = ops.Cast()
        logger.info(
            "Init model: {} --> {}. Model weights loaded from {}".format(
                args.e2e_algorithm, model_name, "pretrained url" if pretrained else ckpt_load_path
            )
        )

        # build preprocess and postprocess
        self.preprocess = Preprocessor(
            task="det",
            algo=args.e2e_algorithm,
            det_limit_side_len=args.e2e_limit_side_len,
            det_limit_type=args.e2e_limit_type,
        )

        self.postprocess = Postprocessor(task="e2e", algo=args.e2e_algorithm)

        self.vis_dir = args.draw_img_save_dir
        os.makedirs(self.vis_dir, exist_ok=True)

        self.box_type = "poly"
        self.visualize_preprocess = True

    def __call__(self, img_or_path, do_visualize=True):
        """
        Args:
            img_or_path: str for img path or np.array for RGB image
            do_visualize: visualize preprocess and final result and save them

        Return:
            e2e_res_final (dict): result with keys:
                - polys: np.array in shape [num_polygons, 4, 2] if e2e_box_type is 'quad'. Otherwise,
                    it is a list of np.array, each np.array is the polygon points.
                - scores: list in shape [num_polygons], confidence of each detected text box.
                - texts: list in shape [num_polygons], text string.
            data (dict): input and preprocessed data with keys: (for visualization and debug)
                - image_ori (np.ndarray): original image in shape [h, w, c]
                - image (np.ndarray): preprocessed image feed for network, in shape [c, h, w]
                - shape (list): shape and scaling information [ori_h, ori_w, scale_ratio_h, scale_ratio_w]
        """
        # preprocess
        data = self.preprocess(img_or_path)

        fn = os.path.basename(data.get("img_path", "input.png")).rsplit(".", 1)[0]
        if do_visualize and self.visualize_preprocess:
            # show_imgs([data['image_ori']], is_bgr_img=False, title='e2e: '+ data['img_path'])
            # TODO: saving images increase inference time.
            show_imgs(
                [data["image"]],
                title=fn + "_e2e_preprocessed",
                mean_rgb=IMAGENET_DEFAULT_MEAN,
                std_rgb=IMAGENET_DEFAULT_STD,
                is_chw=True,
                show=False,
                save_path=os.path.join(self.vis_dir, fn + "_e2e_preproc.png"),
            )
        logger.info(f"Original image shape: {data['image_ori'].shape}")
        logger.info(f"After e2e preprocess: {data['image'].shape}")

        # infer
        input_np = data["image"]
        if len(input_np.shape) == 3:
            net_input = np.expand_dims(input_np, axis=0)

        net_output = self.model(ms.Tensor(net_input))

        # postprocess
        e2e_res = self.postprocess(net_output, data)

        # validate: filter polygons with too small number of points or area
        e2e_res_final = validate_e2e_res(e2e_res, data["image_ori"].shape[:2], min_poly_points=3, min_area=3)

        if do_visualize:
            visualize(data["image_ori"], e2e_res_final["polys"], e2e_res_final["texts"],
                      is_polygon=True, display=False,
                      save_path=os.path.join(self.vis_dir, fn + "_e2e_res.png"),
                      draw_texts_on_blank_page=True)

        return e2e_res_final, data


def validate_e2e_res(e2e_res, img_shape, min_poly_points=3, min_area=3):
    polys = e2e_res["polys"].copy()
    scores = e2e_res.get("scores", [])
    texts = e2e_res["texts"]

    if len(polys) == 0:
        return dict(polys=[], scores=[], texts=[])

    h, w = img_shape[:2]
    # clip if ouf of image
    if not isinstance(polys, list):
        polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
        polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)
    else:
        for i, poly in enumerate(polys):
            polys[i][:, 0] = np.clip(polys[i][:, 0], 0, w - 1)
            polys[i][:, 1] = np.clip(polys[i][:, 1], 0, h - 1)

    new_polys, new_texts = [], []
    if scores is not None:
        new_scores = []
    for i, (poly, text) in enumerate(zip(polys, texts)):
        # filter
        if len(poly) < min_poly_points:
            continue

        if min_area > 0:
            p = Polygon(poly)
            if p.is_valid and not p.is_empty:
                if p.area >= min_area:
                    poly_np = np.array(p.exterior.coords)[:-1, :]
                    new_polys.append(poly_np)
                    new_texts.append(text)
                    if scores is not None:
                        new_scores.append(scores[i])
        else:
            new_polys.append(poly.astype(int))
            new_texts.append(text)
            if scores is not None:
                new_scores.append(scores[i])

    if len(scores) > 0:
        new_e2e_res = dict(polys=new_polys, scores=new_scores, texts=new_texts)
    else:
        new_e2e_res = dict(polys=new_polys, texts=new_texts)

    # TODO: sort polygons from top to bottom, left to right

    return new_e2e_res


def save_e2e_res(e2e_res_all: List[dict], img_paths: List[str], include_score=False, save_path="./e2e_results.txt"):
    lines = []
    for i, e2e_res in enumerate(e2e_res_all):
        if not include_score:
            img_pred = (
                os.path.basename(img_paths[i])
                + "\t"
                + str(json.dumps(e2e_res["texts"]))
                + "\t"
                + str(json.dumps([x.tolist() for x in e2e_res["polys"]]))
                + "\n"
            )
        else:
            img_pred = (
                os.path.basename(img_paths[i])
                + "\t"
                + str(json.dumps(e2e_res["texts"]))
                + "\t"
                + str(json.dumps(e2e_res["scores"]))
                + "\t"
                + str(json.dumps([x.tolist() for x in e2e_res["polys"]]))
                + "\n"
            )
        lines.append(img_pred)

    with open(save_path, "w") as f:
        f.writelines(lines)
        f.close()


if __name__ == "__main__":
    # parse args
    args = parse_args()
    set_logger(name="mindocr")
    save_dir = args.draw_img_save_dir
    img_paths = get_image_paths(args.image_dir)
    # uncomment it to quick test the infer FPS
    # img_paths = img_paths[:15]

    ms.set_context(mode=args.mode)

    # init detector
    text_detect = TextEnd2End(args)

    # run for each image
    e2e_res_all = []
    for i, img_path in enumerate(img_paths):
        logger.info(f"\nInfering [{i+1}/{len(img_paths)}]: {img_path}")
        e2e_res, _ = text_detect(img_path, do_visualize=True)
        e2e_res_all.append(e2e_res)
        logger.info(f"Num detected text boxes: {len(e2e_res['polys'])}")

    # save all results in a txt file
    save_e2e_res(e2e_res_all, img_paths, save_path=os.path.join(save_dir, "e2e_results.txt"))

    logger.info(f"Done! Text detection & recognition results saved in {save_dir}")
