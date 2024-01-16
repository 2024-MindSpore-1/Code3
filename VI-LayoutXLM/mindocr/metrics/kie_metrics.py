from seqeval.metrics import f1_score, precision_score, recall_score

import mindspore as ms
from mindspore import Tensor

from ..utils.misc import AllReduce

__all__ = ["VQASerTokenMetric"]

class VQASerTokenMetric:
    def __init__(self, device_num=1, **kwargs):
        self.clear()
        self.device_num = device_num
        self.all_reduce = AllReduce(reduce="sum") if device_num > 1 else None
        self.metric_names = ["precision", "recall", "hmean"]

    def clear(self):
        self.pred_list = []
        self.gt_list = []

    def update(self, *inputs):
        if len(inputs) != 2:
            raise ValueError("Length of inputs should be 2")
        post_res, _ = inputs
        preds = post_res["decode_out"]
        labels = post_res["label_decode_out"]
        self.pred_list.extend(preds)
        self.gt_list.extend(labels)

    def eval(self):
        precision = Tensor(precision_score(self.gt_list, self.pred_list), dtype=ms.float32)
        recall = Tensor(recall_score(self.gt_list, self.pred_list), dtype=ms.float32)
        hmean = Tensor(f1_score(self.gt_list, self.pred_list), dtype=ms.float32)
        if self.all_reduce:
            precision = float((self.all_reduce(precision) / self.device_num).asnumpy())
            recall = float((self.all_reduce(recall) / self.device_num).asnumpy())
            hmean = float((self.all_reduce(hmean) / self.device_num).asnumpy())
        else:
            precision = float(precision.asnumpy())
            recall = float(recall.asnumpy())
            hmean = float(hmean.asnumpy())
        metrics = {
            "precision": precision,
            "recall": recall,
            "hmean": hmean
        }
        return metrics
