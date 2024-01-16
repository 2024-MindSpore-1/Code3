import argparse
import pickle as pkl

import numpy as np
import scanpy as sc
import mindspore as ms
from mindspore import Tensor, load_checkpoint, nn, ops

from performer_mindspore import PerformerLM

ms.set_context(device_target='GPU')

parser = argparse.ArgumentParser()
parser.add_argument("--bin_num", type=int, default=5, help='Number of bins.')
parser.add_argument("--gene_num", type=int, default=16906, help='Number of genes.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--novel_type", type=bool, default=False, help='Novel cell tpye exists or not.')
parser.add_argument("--unassign_thres", type=float, default=0.5, help='The confidence score threshold for novel cell type annotation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for predicting.')
parser.add_argument("--model_path", type=str, default='./scbert.ckpt', help='Path of finetuned model.')

args = parser.parse_args()

SEED = args.seed
EPOCHS = args.epoch
SEQ_LEN = args.gene_num + 1
UNASSIGN = args.novel_type
UNASSIGN_THRES = args.unassign_thres if UNASSIGN == True else 0
CLASS = args.bin_num + 2
POS_EMBED_USING = args.pos_embed

class Identity(nn.Cell):
    def __init__(self, dropout=0, h_dim=100, out_dim=10):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200),pad_mode='valid', has_bias=True)
        self.act = nn.ReLU()
        self.fc1 = nn.Dense(in_channels=SEQ_LEN, out_channels=512, has_bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(1.0-dropout)
        self.fc2 = nn.Dense(in_channels=512, out_channels=h_dim, has_bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(1.0-dropout)
        self.fc3 = nn.Dense(in_channels=h_dim, out_channels=out_dim, has_bias=True)

    def construct(self, x):
        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

data = sc.read_h5ad(args.data_path)
label_dict, label = np.unique(np.array(data.obs['celltype']), return_inverse=True)
class_num = np.unique(label, return_counts=True)[1].tolist()
data = data.X

model = PerformerLM(
    num_tokens = CLASS,
    dim = 200,
    depth = 6,
    max_seq_len = SEQ_LEN,
    heads = 10,
    local_attn_heads = 0,
)
model.to_out = Identity(dropout=0., h_dim=128, out_dim=label_dict.shape[0])

path = args.model_path
ms.load_checkpoint(path, model)

batch_size = data.shape[0]
model.set_train(False)
pred_finals = []
novel_indices = []
for index in range(batch_size):
    full_seq = data[index].toarray()[0]
    full_seq[full_seq > (CLASS - 2)] = CLASS - 2
    full_seq = Tensor(full_seq, ms.int32)
    full_seq = ops.cat((full_seq, Tensor([0], ms.int32)))
    full_seq = full_seq.unsqueeze(0)
    pred_logits = model(full_seq)
    softmax = nn.Softmax(axis=-1)
    pred_prob = softmax(pred_logits)
    pred_final = pred_prob.argmax(axis=-1)
    if np.amax(pred_prob.asnumpy(), axis=-1) < UNASSIGN_THRES:
        novel_indices.append(index)
    pred_finals.append(pred_final.asnumpy().tolist())
pred_list = label_dict[pred_finals].tolist()
for index in novel_indices:
    pred_list[index] = 'Unassigned'
print(pred_list)