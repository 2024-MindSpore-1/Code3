from mindspore import nn, ops
from mindspore.nn.loss.loss import LossBase

__all__ = ["VQASerTokenLayoutLMLoss"]

class VQASerTokenLayoutLMLoss(LossBase):
    def __init__(self, num_classes, **kwargs):
        super(VQASerTokenLayoutLMLoss, self).__init__()
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss(reduction="none")

    def construct(self, predicts, attention_mask, labels):
        predicts = ops.reshape(predicts, (-1, self.num_classes))
        labels = ops.reshape(labels, (-1,))
        if attention_mask is not None:
            attention_mask = ops.reshape(attention_mask, (-1,))
            loss = self.loss(predicts, labels)
            loss = (loss * attention_mask).sum() / (attention_mask.sum() + 1e-5)
        else:
            loss = self.loss(predicts, labels).mean()
        return loss
