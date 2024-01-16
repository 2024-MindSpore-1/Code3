# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import datetime
import os
import math
from functools import partial

import numpy as np
from mindspore import Tensor, nn, Parameter, ops
from mindspore import dtype as mstype
from mindspore.common import initializer as weight_init
from mindspore.common.initializer import initializer, Uniform


class PositionwiseFeedforward(nn.Cell):
    

    def __init__(self, hid_dim, pf_dim, dropout):

        super(PositionwiseFeedforward, self).__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units

        self.do = nn.Dropout(dropout)


    def construct(self, x):
        # x = [batch size, sent len, hid dim]

        x = x.permute(0, 2, 1)

        # x = [batch size, hid dim, sent len]

        x = self.do(ops.relu(self.fc_1(x)))

        # x = [batch size, pf dim, sent len]

        x = self.fc_2(x)

        # x = [batch size, hid dim, sent len]

        x = x.permute(0, 2, 1)

        # x = [batch size, sent len, hid dim]

        return x  

class DecoderLayer(nn.Cell):
    

    def __init__(self, hid_dim, n_heads, pf_dim, dropout):

        super(DecoderLayer, self).__init__()
        self.ln = nn.LayerNorm((hid_dim,))
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)



    def construct(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))

        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))

        trg = self.ln(trg + self.do(self.pf(trg)))

        return trg
    
class SelfAttention(nn.Cell):
    

    def __init__(self, hid_dim, n_heads, dropout):

        super(SelfAttention, self).__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Dense(hid_dim, hid_dim)
        self.w_k = nn.Dense(hid_dim, hid_dim)
        self.w_v = nn.Dense(hid_dim, hid_dim)

        self.fc = nn.Dense(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = ops.Sqrt()(Tensor([hid_dim // n_heads], dtype=mstype.float16))


    def construct(self, query, key, value, mask=None):
        bsz = query.shape[0]

        # query = key = value [batch size, sent len, hid dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # Q, K, V = [batch size, sent len, hid dim]

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        # K, V = [batch size, n heads, sent len_K, hid dim // n heads]
        # Q = [batch size, n heads, sent len_q, hid dim // n heads]
        #energy = ops.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        energy = ops.bmm(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = [batch size, n heads, sent len_Q, sent len_K]
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(ops.softmax(energy, axis=-1))

        # attention = [batch size, n heads, sent len_Q, sent len_K]

        #x = ops.matmul(attention, V)
        x = ops.bmm(attention, V)
        # x = [batch size, n heads, sent len_Q, hid dim // n heads]

        x = x.permute(0, 2, 1, 3)     #.contiguous()

        # x = [batch size, sent len_Q, n heads, hid dim // n heads]

        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch size, src sent len_Q, hid dim]

        x = self.fc(x)

        # x = [batch size, sent len_Q, hid dim]
        
        return x
        
    
class Encoder(nn.Cell):
    

    def __init__(self, protein_dim, hid_dim, n_layers,kernel_size , dropout):

        super(Encoder, self).__init__()

        assert kernel_size % 2 == 1, "Kernel size must be odd (for now)"

        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
   
        self.scale = ops.Sqrt()(Tensor([0.5], dtype=mstype.float16))


        self.conv_1 = nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, pad_mode='pad',  padding=(kernel_size-1)//2)
        self.conv_2 = nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, pad_mode='pad',  padding=(kernel_size-1)//2)
        self.conv_3 = nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, pad_mode='pad',  padding=(kernel_size-1)//2)
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Dense(self.input_dim, self.hid_dim)

    def construct(self, protein):
        
        
        conv_input = self.fc(protein)
        conv_input = ops.Transpose()(conv_input, (0, 2, 1))
        
        conved = self.conv_1(self.dropout(conv_input))
        conved = ops.glu(conved, axis=1)
        conved = (conved + conv_input) * self.scale
        conv_input = conved
        conved = self.conv_2(self.dropout(conv_input))
        conved = ops.glu(conved, axis=1)
        conved = (conved + conv_input) * self.scale
        conv_input = conved
        conved = self.conv_3(self.dropout(conv_input))
        conved = ops.glu(conved, axis=1)
        conved = (conved + conv_input) * self.scale
        conv_input = conved
        
        
        
        conved = ops.Transpose()(conved, (0, 2, 1))
        
       
        
        # conved = [batch size,protein len,hid dim]
        return conved
        #return conv_input
        
class Decoder(nn.Cell):
    

    def __init__(self, atom_dim, hid_dim, n_layers, n_heads, pf_dim, dropout):

        super(Decoder, self).__init__()
        #self.ln = nn.LayerNorm((hid_dim,))
        self.output_dim = atom_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim

        self.dropout = dropout
 
        #self.sa = SelfAttention(hid_dim, n_heads, dropout) #unused

        self.layer_1 = DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
        self.layer_2 = DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
        self.layer_3 = DecoderLayer(hid_dim, n_heads, pf_dim, dropout)
        self.ft = nn.Dense(atom_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.fc_1 = nn.Dense(hid_dim, 256)
        self.fc_2 = nn.Dense(256, 2)
        self.batch_size = 32

    def construct(self, trg, src, trg_mask=None,src_mask=None):
        
        trg = self.ft(trg)
        
        trg = self.layer_1(trg, src)
        trg = self.layer_2(trg, src)
        trg = self.layer_3(trg, src)
        
        # default value?
        norm = ops.norm(trg, axis=2)
        
        # norm = [batch size,compound len]
        norm = ops.softmax(norm, axis=1)
        # norm = [batch size,compound len]
        #trg = ops.squeeze(trg, axis=0)
        #norm = ops.squeeze(norm, axis=0)
        
        sum = Tensor(np.zeros([self.batch_size, self.hid_dim]), dtype=mstype.float16)
        
        for j in range(self.batch_size):
            trg_temp = trg[j]
            norm_temp = norm[j]
            for i in range(norm_temp.shape[0]):
                v = trg_temp[i,]
                v = v * norm_temp[i]
                sum[j] += v
        
        #sum = ops.expand_dims(sum,  0)
        
        # trg = [batch size,hid_dim]
        label = ops.ReLU()(self.fc_1(sum))
        label = self.fc_2(label)

        return label


class TransformerCPI(nn.Cell):

    def __init__(self, atom_dim=34):
        # TransformerCPI
        super().__init__()
        
        self.encoder = Encoder(protein_dim=100, hid_dim=64, n_layers=3, kernel_size=5, dropout=0.1)
        self.decoder = Decoder(atom_dim=34, hid_dim=64, n_layers=3, n_heads=8, pf_dim=256, dropout=0.1)
        stdv = 1. / math.sqrt(atom_dim)
        self.weight = Parameter(initializer(Uniform(scale=stdv), [atom_dim, atom_dim], mstype.float16))


    def gcn(self, compound, adj):
        
        
        #support = ops.matmul(compound, self.weight)
        support = ops.matmul(compound, self.weight)

        #output = ops.matmul(adj, support)
        output = ops.bmm(adj, support)
        return output

    def construct(self, compound, adj, protein):
        
        compound = self.gcn(compound, adj)
        
        
        
        #compound = ops.ExpandDims()(compound, 0)
        #protein = ops.ExpandDims()(protein, 0)
        
        
        enc_src = self.encoder(protein)
        
        out = self.decoder(compound, enc_src)
        
        return out


#def TransformerCPI():
#    return TransformerCPI(atom_dim=34)



