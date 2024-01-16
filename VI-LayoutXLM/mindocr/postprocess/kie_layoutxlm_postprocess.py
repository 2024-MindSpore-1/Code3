from mindspore import Tensor

from ..data.transforms.kie_transforms import load_vqa_bio_label_maps

__all__ = ["VQASerTokenLayoutLMPostProcess"]

class VQASerTokenLayoutLMPostProcess:
    def __init__(self, class_path, **kwargs):
        label2id_map, self.id2label_map = load_vqa_bio_label_maps(class_path)

        self.label2id_map_for_draw = dict()
        for key in label2id_map:
            if key.startswith("I-"):
                self.label2id_map_for_draw[key] = label2id_map["B" + key[1:]]
            else:
                self.label2id_map_for_draw[key] = label2id_map[key]

        self.id2label_map_for_show = dict()
        for key in self.label2id_map_for_draw:
            val = self.label2id_map_for_draw[key]
            if key == "O":
                self.id2label_map_for_show[val] = key
            if key.startswith("B-") or key.startswith("I-"):
                self.id2label_map_for_show[val] = key[2:]
            else:
                self.id2label_map_for_show[val] = key

    def _eval(self, preds, labels):
        pred_idxs = preds.argmax(axis=2)
        decode_out_list = [[] for _ in range(pred_idxs.shape[0])]
        label_decode_out_list = [[] for _ in range(pred_idxs.shape[0])]

        for i in range(pred_idxs.shape[0]):
            for j in range(pred_idxs.shape[1]):
                if labels[i, j] != -100:
                    decode_out_list[i].append(self.id2label_map[pred_idxs[i, j]])
                    label_decode_out_list[i].append(self.id2label_map[labels[i, j]])

        return {"decode_out": decode_out_list, "label_decode_out": label_decode_out_list}

    def __call__(self, preds, labels, **kwargs):
        labels = labels[1]
        if isinstance(preds, Tensor):
            preds = preds.asnumpy()
            labels = labels.asnumpy()

        return self._eval(preds, labels)
