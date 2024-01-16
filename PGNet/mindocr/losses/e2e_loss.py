import mindspore as ms
import mindspore.numpy as mnp
from mindspore import Tensor, nn, ops

from .det_loss import DiceLoss

__all__ = ["PGLoss"]


def dummy(*args, **kwargs):
    return


class PGLoss(nn.LossBase):
    """
    text center line（TCL）-（文本中心线检测）
    text border offset（TBO）-（文本边框偏移）
    text direction offset（TDO）-（文本方向偏移）
    text character classification（TCC）-文本字符分类

    Args:
        tcl_len (int): 一个text的tcl的像素长度.
        max_text_len (int): 一个text有多长.
        max_text_nums (int): 一张图片有几个text.
        eps (float, optional): _description_. Defaults to 1e-6.
    """

    def __init__(self, 
                 tcl_len: int, 
                 max_text_len: int, 
                 max_text_nums: int, 
                 eps: float = 1e-6, 
                 debug: bool = False, 
                 **kwargs):
        super().__init__(reduction="sum")
        self.tcl_len = tcl_len
        self.max_text_nums = max_text_nums
        self.max_text_len = max_text_len

        # NOTE DEBUG, only graph mode
        self.loss_summary = ops.ScalarSummary() if ms.get_context('mode') == ms.GRAPH_MODE and debug else dummy

        self.sequence_length = mnp.array([tcl_len] * max_text_nums, dtype=ms.int32)
        label_indices = []
        for i in range(max_text_nums):
            for j in range(max_text_len):
                label_indices.append([i, j])
        self.label_indices = mnp.array(label_indices, dtype=ms.int64)
        self.ctc_loss = ops.CTCLoss(ctc_merge_repeated=True)
        self.dice_loss = DiceLoss(eps=eps)

    def smooth_l1_loss(self, pred: Tensor, label: Tensor, mask: Tensor) -> Tensor:
        label, norm = label[:, :-1], label[:, -1:]

        abs_diff = mnp.abs(label - pred)
        loss_sign = ops.stop_gradient(abs_diff < 1)
        weight = norm * (0.5 * ops.pow(abs_diff, 2) * loss_sign + (abs_diff - 0.5) * (~loss_sign))

        loss = mnp.sum(weight * mask) / (mnp.sum(mask) * label.shape[1] + 1e-5)  # NOTE 这里有坑，要乘label.shape[1]
        return loss

    def tcc_loss(self, f_char: Tensor, ctc_points: Tensor, ctc_masks: Tensor, ctc_labels: Tensor) -> Tensor:
        """TCC loss, text character classification（TCC）-文本字符分类"""
        f_char = f_char.transpose(0, 2, 3, 1)  # BCHW -> BHWC
        ctc_labels = ctc_labels.reshape(ctc_labels.shape[0], -1)

        # 原paddle ocr限制batch内总text个数的操作很迷惑
        total_loss, avg_factor = 0.0, 0.0
        for i, pred in enumerate(f_char):
            points = ctc_points[i]
            mask   = ctc_masks [i]
            labels = ctc_labels[i]

            label_weight = mask.any(-1).astype(f_char.dtype)
            avg_factor = avg_factor + label_weight.sum()

            logit = ops.gather_nd(pred, points)  # [self.max_text_nums, self.tcl_len, C]
            logit = logit.reshape(self.max_text_nums, self.tcl_len, pred.shape[-1])  # 动态shape转静态shape

            mask = mask[..., None].astype(f_char.dtype)
            logit_ld = logit * mask   # [self.max_text_nums, self.max_text_len, 37]
            weight = (1 - mask) * 20  # [self.max_text_nums, self.max_text_len,  1]  # FIXME 填充的置20，算是一种smooth吗？
            logit_ld[..., :-1] -= weight
            logit_ld[..., -1:] += weight
            logit_ld = logit_ld.transpose(1, 0, 2)  # [self.max_text_len, self.max_text_nums, 37]

            loss, _ = self.ctc_loss(logit_ld, self.label_indices, labels, self.sequence_length)  # [self.max_text_nums]
            loss = self.get_loss(loss, label_weight)
            total_loss = total_loss + loss

        final_loss = total_loss / avg_factor
        return final_loss

    def construct(
        self,
        preds: tuple,
        tcl_maps: Tensor,
        tbo_maps: Tensor,
        tdo_maps: Tensor,
        training_masks: Tensor,  # 排除非法字符区域
        ctc_points: Tensor,
        ctc_masks: Tensor,
        ctc_labels: Tensor,
    ) -> Tensor:
        """_summary_

        Args:
            preds (tuple): _description_
            tcl_maps (Tensor[float]): [b, 1, h, w]
            tbo_maps (Tensor[float]): [b, 5, h, w]
            tdo_maps (Tensor[float]): [b, 3, h, w]
            training_masks (Tensor[float]): [b, 1, h, w]
            ctc_masks  (Tensor[bool ]): [b, max_text_num, max_text_len]
            ctc_labels (Tensor[int32]): [b, max_text_num, max_text_len]

        Returns:
            Tensor: _description_
        """
        f_score, f_border, f_char, f_direction = preds
        masked_tcl_maps = tcl_maps * training_masks

        tcl_loss = self.dice_loss(f_score, tcl_maps, training_masks.squeeze(axis=1))
        tbo_loss = self.smooth_l1_loss(f_border   , tbo_maps, masked_tcl_maps)
        tdo_loss = self.smooth_l1_loss(f_direction, tdo_maps, masked_tcl_maps)
        tcc_loss = self.tcc_loss(f_char, ctc_points, ctc_masks, ctc_labels)

        loss = tcl_loss + tbo_loss + tdo_loss + tcc_loss * 5  # FIXME *5 可以设置为loss_weight

        # NOTE DEBUG, only graph mode
        self.loss_summary('loss', loss)
        self.loss_summary('tcl_loss', tcl_loss)
        self.loss_summary('tbo_loss', tbo_loss)
        self.loss_summary('tdo_loss', tdo_loss)
        self.loss_summary('tcc_loss', tcc_loss)
        return loss
