from typing import List
import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

from mindocr.metrics.e2e_metrics import E2EMetric

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "../..")))


def expand_points(boxes: list) -> list:
    if len(boxes) == 0:
        return boxes

    max_points_num = max(len(b) for b in boxes)
    ex_boxes = [b + [b[-1]] * (max_points_num - len(b)) for b in boxes]  # NOTE 允许奇数
    return ex_boxes


def _e2e_adapt_train_pred(content: List[dict]) -> dict:
    boxes, scores, texts = [], [], []
    for con in content:
        boxes.append(con["points"])
        scores.append(con["score"])
        texts.append(con["transcription"])

    boxes = np.array(boxes)[None]
    scores = np.array(scores)[None]
    return {"polys": boxes, "scores": scores, "texts": [texts]}


def _e2e_adapt_train_label(content: List[dict], img_id: int) -> List[np.ndarray]:
    boxes, texts, ignored_tag = [], [], []
    for con in content:
        boxes.append(con["points"])
        text = con["transcription"].lower()  # NOTE 包含模型无法处理的特殊字符
        texts.append(text)
        ignored_tag.append(text in ("###", "*"))

    boxes = np.array(expand_points(boxes))[None]
    ignored_tag = np.array(ignored_tag).reshape(1, -1)
    return [[img_id], boxes, [texts], ignored_tag]


def eval_e2e_adapt_train(preds: dict, labels: dict) -> dict:
    metric = E2EMetric(character_dict_path="mindocr/utils/dict/ic15_dict.txt", mode="a")
    # metric = E2EMetric(character_dict_path="mindocr/utils/dict/ic15_dict.txt",
    #                    mode='b', gt_mat_dir="data/total_text/test/Groundtruth")

    for img_name in tqdm(preds.keys()):
        pred = _e2e_adapt_train_pred(preds[img_name])
        label = _e2e_adapt_train_label(labels[img_name], int(img_name[3:-4]))
        metric.update(pred, label)

    eval_res = metric.eval()
    return eval_res


def read_content(filename: str, is_gt: bool = False) -> dict:
    results = {}
    with open(filename, encoding="utf-8") as f:
        for line in f:
            name, content = line.split("\t", 1)
            if is_gt:
                name = os.path.basename(name)
            results[name] = json.loads(content)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", required=True, type=str)
    parser.add_argument("--pred_path", required=True, type=str)
    args = parser.parse_args()

    labels = read_content(args.gt_path, is_gt=True)
    preds = read_content(args.pred_path)
    labels_keys = labels.keys()
    preds_keys = preds.keys()

    if set(labels_keys) != set(preds_keys):
        raise ValueError("The images in gt_path and pred_path must be the same.")

    print("----- Start adapted eval e2e------")
    eval_res = eval_e2e_adapt_train(preds, labels)
    print(eval_res)
    print("----- End adapted eval e2e------")
