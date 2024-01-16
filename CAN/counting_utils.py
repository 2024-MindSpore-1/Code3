import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import numpy as np
import matplotlib.pyplot as plt
import os

def gen_counting_label(labels, channel, tag):
    b, t = labels.shape
    labels_numpy = labels.asnumpy()
    counting_labels = np.zeros((b,channel))
    if tag:
        ignore = [0, 1, 107, 108, 109, 110]
    else:
        ignore = []
    for i in range(b):
        for j in range(t):
            k = labels_numpy[i][j]
            if k in ignore:
                continue
            else:
                counting_labels[i][k] += 1

    return ms.Tensor(counting_labels,ms.float32)
