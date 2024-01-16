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
"""train"""
import numpy as np
import scipy.sparse as sp
import datetime
import os
import mindspore as ms
from mindspore import Model
from mindspore import context
from mindspore import nn
from mindspore.common import set_seed
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore import Tensor
from mindspore import amp, nn
from mindspore import ops
from mindspore.common import dtype as mstype

from src.args import args
from src.tools.callback import EvaluateCallBack
from src.tools.cell import cast_amp
from src.tools.criterion import NetWithLoss
from src.tools.get_misc import get_dataset, set_device, get_model, get_pretrained, get_train_one_step
from src.tools.optimizer import get_optimizer
from mindspore import Profiler
from mindspore.train import ROC, auc
from sklearn.metrics import average_precision_score



def _bmat_fast(mats):
    new_data = np.concatenate([mat.data for mat in mats])

    ind_offset = np.zeros(1 + len(mats))
    ind_offset[1:] = np.cumsum([mat.shape[0] for mat in mats])
    new_indices = np.concatenate(
        [mats[i].indices + ind_offset[i] for i in range(len(mats))])

    indptr_offset = np.zeros(1 + len(mats))
    indptr_offset[1:] = np.cumsum([mat.nnz for mat in mats])
    new_indptr = np.concatenate(
        [mats[i].indptr[i >= 1:] + indptr_offset[i] for i in range(len(mats))])
    return sp.csr_matrix((new_data, new_indices, new_indptr))
    
def main():

    set_seed(args.seed)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    #context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    context.set_context(enable_graph_kernel=False)
    #context.set_context(save_graphs=True)
    #if args.device_target == "Ascend":
    #    context.set_context(enable_auto_mixed_precision=True)
    rank = set_device(args)
    # get model and cast amp_level
    #profiler = Profiler()
    
    batch_size = args.batch_size

    net = get_model(args)
    net = amp.auto_mixed_precision(net, "O0")
    #cast_amp(net)
    #net = net.to_float(ms.float16)
    
    net_with_loss = NetWithLoss(net)
    if args.pretrained:
        get_pretrained(args, net)

    data = get_dataset(args)
    #batch_num = data.train_dataset.get_dataset_size()
    
    

    batch_num = 110000 // batch_size
    id = data.id
    N = data.N
    Z = data.Z
    R = data.R
    targets = data.targets
    N_cumsum = data.N_cumsum

    all_idx = np.arange(len(id))
    all_idx = np.random.RandomState(seed=42).permutation(all_idx)
    train_idx = all_idx[0:110000]
    valid_idx = all_idx[110000:120000]
    test_idx = all_idx[120000:]
    print('=========idx_length==========')
    print(len(train_idx))
    print(len(valid_idx))
    print(len(test_idx))
    train_idx = np.random.RandomState(seed=42).permutation(train_idx)
    print(all_idx)
    print(train_idx)
    print(valid_idx)
    print(test_idx)

    optimizer = get_optimizer(args, net, batch_num)


    train_net = get_train_one_step(args, net_with_loss, optimizer)

    #metric = ROC()
    '''  real_value
    Z_max = 803
    id3dnb_i_max = 391640
    idnb_i_max = 17786
    '''
    Z_max = 810
    id3dnb_i_max = 392000
    idnb_i_max = 17800
        
    for epoch in range(args.epochs):
        print('========epoch=============')
        print(epoch)

        train_net.set_train(mode=True)
        
        for i in range(110000 // batch_size):
            
            idx_batch = train_idx[i*batch_size:(i+1)*batch_size]
            #train_idx_batch = train_idx[0*batch_size:(0+1)*batch_size]
            data = {}
            data['targets'] = targets[idx_batch]

            data['id'] = id[idx_batch]

            data['N'] = N[idx_batch]
            
            data['batch_seg'] = np.repeat(np.arange(len(idx_batch), dtype=np.int32), data['N'])

            adj_matrices = []
            
            data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
            data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)
            
            nend = 0
            for k, i in enumerate(idx_batch):
                n = data['N'][k]  # number of atoms
                nstart = nend
                nend = nstart + n

                if Z is not None:
                    data['Z'][nstart:nend] = Z[N_cumsum[i]:N_cumsum[i + 1]]

                R_temp = R[N_cumsum[i]:N_cumsum[i + 1]]
                data['R'][nstart:nend] = R_temp

                Dij = np.linalg.norm(R_temp[:, None, :] - R_temp[None, :, :], axis=-1)
                adj_matrices.append(sp.csr_matrix(Dij <= 5.0)) # cutoff = 5.0
                adj_matrices[-1] -= sp.eye(n, dtype=np.bool)

            
            adj_matrix = _bmat_fast(adj_matrices)
            
            atomids_to_edgeid = sp.csr_matrix((np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr), shape=adj_matrix.shape)
            
            edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()
            
            # Target (i) and source (j) nodes of edges
            data['idnb_i'] = edgeid_to_target
            data['idnb_j'] = edgeid_to_source
            
            # Indices of triplets k->j->i
            ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
            id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
            id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
            id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]
            
            
            # Indices of triplets that are not i->j->i
            id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
            data['id3dnb_i'] = id3ynb_i[id3_y_to_d]
            data['id3dnb_j'] = id3ynb_j[id3_y_to_d]
            data['id3dnb_k'] = id3ynb_k[id3_y_to_d]
            
            # Edge indices for interactions
            # j->i => k->j
            data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
            # j->i => k->j => j->i
            data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]

            
            data['Z'] = np.pad(data['Z'], ((0, Z_max-data['Z'].shape[0])), 'constant', constant_values=0)
            
            data['R'] = np.pad(data['R'], ((0, Z_max-data['R'].shape[0]), (0, 0)), 'constant', constant_values=0.0)
            data['batch_seg'] = np.pad(data['batch_seg'], ((0, Z_max-data['batch_seg'].shape[0])), 'constant', constant_values=batch_size)
            data['idnb_i'] = np.pad(data['idnb_i'], ((0, idnb_i_max-data['idnb_i'].shape[0])), 'constant', constant_values=Z_max-1)
            data['idnb_j'] = np.pad(data['idnb_j'], ((0, idnb_i_max-data['idnb_j'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id_expand_kj'] = np.pad(data['id_expand_kj'], ((0, id3dnb_i_max-data['id_expand_kj'].shape[0])), 'constant', constant_values=idnb_i_max-1)
            data['id_reduce_ji'] = np.pad(data['id_reduce_ji'], ((0, id3dnb_i_max-data['id_reduce_ji'].shape[0])), 'constant', constant_values=idnb_i_max-1)
            data['id3dnb_i'] = np.pad(data['id3dnb_i'], ((0, id3dnb_i_max-data['id3dnb_i'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id3dnb_j'] = np.pad(data['id3dnb_j'], ((0, id3dnb_i_max-data['id3dnb_j'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id3dnb_k'] = np.pad(data['id3dnb_k'], ((0, id3dnb_i_max-data['id3dnb_k'].shape[0])), 'constant', constant_values=Z_max-1)
            
           
            loss = train_net(Tensor(data['Z']), Tensor(data['R']), Tensor(data['batch_seg']), Tensor(data['idnb_i']), Tensor(data['idnb_j']), Tensor(data['id_expand_kj'], dtype=mstype.int32), Tensor(data['id_reduce_ji']), Tensor(data['id3dnb_i']), Tensor(data['id3dnb_j']), Tensor(data['id3dnb_k']), Tensor(data['targets']))
            
        
        
            #profiler.analyse()
        
        # validation
        net.set_train(False)
        total_loss_valid = 0.0
        for i in range(10000 // batch_size):
            
            

            idx_batch = valid_idx[i*batch_size:(i+1)*batch_size]

            data = {}
            data['targets'] = targets[idx_batch]

            data['id'] = id[idx_batch]

            data['N'] = N[idx_batch]
            
            data['batch_seg'] = np.repeat(np.arange(len(idx_batch), dtype=np.int32), data['N'])

            adj_matrices = []
            
            data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
            data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)
            
            nend = 0
            for k, i in enumerate(idx_batch):
                n = data['N'][k]  # number of atoms
                nstart = nend
                nend = nstart + n

                if Z is not None:
                    data['Z'][nstart:nend] = Z[N_cumsum[i]:N_cumsum[i + 1]]

                R_temp = R[N_cumsum[i]:N_cumsum[i + 1]]
                data['R'][nstart:nend] = R_temp

                Dij = np.linalg.norm(R_temp[:, None, :] - R_temp[None, :, :], axis=-1)
                adj_matrices.append(sp.csr_matrix(Dij <= 5.0)) # cutoff = 5.0
                adj_matrices[-1] -= sp.eye(n, dtype=np.bool)

            
            adj_matrix = _bmat_fast(adj_matrices)
            
            atomids_to_edgeid = sp.csr_matrix((np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr), shape=adj_matrix.shape)
            
            edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()
            
            # Target (i) and source (j) nodes of edges
            data['idnb_i'] = edgeid_to_target
            data['idnb_j'] = edgeid_to_source
            
            # Indices of triplets k->j->i
            ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
            id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
            id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
            id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]
            
            
            # Indices of triplets that are not i->j->i
            id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
            data['id3dnb_i'] = id3ynb_i[id3_y_to_d]
            data['id3dnb_j'] = id3ynb_j[id3_y_to_d]
            data['id3dnb_k'] = id3ynb_k[id3_y_to_d]
            
            # Edge indices for interactions
            # j->i => k->j
            data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
            # j->i => k->j => j->i
            data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]

            
            data['Z'] = np.pad(data['Z'], ((0, Z_max-data['Z'].shape[0])), 'constant', constant_values=0)
            data['R'] = np.pad(data['R'], ((0, Z_max-data['R'].shape[0]), (0, 0)), 'constant', constant_values=0.0)
            data['batch_seg'] = np.pad(data['batch_seg'], ((0, Z_max-data['batch_seg'].shape[0])), 'constant', constant_values=batch_size)
            data['idnb_i'] = np.pad(data['idnb_i'], ((0, idnb_i_max-data['idnb_i'].shape[0])), 'constant', constant_values=Z_max-1)
            data['idnb_j'] = np.pad(data['idnb_j'], ((0, idnb_i_max-data['idnb_j'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id_expand_kj'] = np.pad(data['id_expand_kj'], ((0, id3dnb_i_max-data['id_expand_kj'].shape[0])), 'constant', constant_values=idnb_i_max-1)
            data['id_reduce_ji'] = np.pad(data['id_reduce_ji'], ((0, id3dnb_i_max-data['id_reduce_ji'].shape[0])), 'constant', constant_values=idnb_i_max-1)
            data['id3dnb_i'] = np.pad(data['id3dnb_i'], ((0, id3dnb_i_max-data['id3dnb_i'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id3dnb_j'] = np.pad(data['id3dnb_j'], ((0, id3dnb_i_max-data['id3dnb_j'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id3dnb_k'] = np.pad(data['id3dnb_k'], ((0, id3dnb_i_max-data['id3dnb_k'].shape[0])), 'constant', constant_values=Z_max-1)
            

            preds = net(Tensor(data['Z']), Tensor(data['R']), Tensor(data['batch_seg']), Tensor(data['idnb_i']), Tensor(data['idnb_j']), Tensor(data['id_expand_kj'], dtype=mstype.int32), Tensor(data['id_reduce_ji']), Tensor(data['id3dnb_i']), Tensor(data['id3dnb_j']), Tensor(data['id3dnb_k']))
            preds = preds[0:batch_size]
            loss = ops.ReduceMean()(ops.abs(Tensor(data['targets']) - preds))
            total_loss_valid = total_loss_valid + loss / (10000 // batch_size)
        print('============loss_valid==========')
        print(total_loss_valid)
        # test
        total_loss_test = 0
        for i in range((len(all_idx) - 120000) // batch_size):
            
            idx_batch = test_idx[i*batch_size:(i+1)*batch_size]

            data = {}
            data['targets'] = targets[idx_batch]

            data['id'] = id[idx_batch]

            data['N'] = N[idx_batch]
            
            data['batch_seg'] = np.repeat(np.arange(len(idx_batch), dtype=np.int32), data['N'])

            adj_matrices = []
            
            data['Z'] = np.zeros(np.sum(data['N']), dtype=np.int32)
            data['R'] = np.zeros([np.sum(data['N']), 3], dtype=np.float32)
            
            nend = 0
            for k, i in enumerate(idx_batch):
                n = data['N'][k]  # number of atoms
                nstart = nend
                nend = nstart + n

                if Z is not None:
                    data['Z'][nstart:nend] = Z[N_cumsum[i]:N_cumsum[i + 1]]

                R_temp = R[N_cumsum[i]:N_cumsum[i + 1]]
                data['R'][nstart:nend] = R_temp

                Dij = np.linalg.norm(R_temp[:, None, :] - R_temp[None, :, :], axis=-1)
                adj_matrices.append(sp.csr_matrix(Dij <= 5.0)) # cutoff = 5.0
                adj_matrices[-1] -= sp.eye(n, dtype=np.bool)

            
            adj_matrix = _bmat_fast(adj_matrices)
            
            atomids_to_edgeid = sp.csr_matrix((np.arange(adj_matrix.nnz), adj_matrix.indices, adj_matrix.indptr), shape=adj_matrix.shape)
            
            edgeid_to_target, edgeid_to_source = adj_matrix.nonzero()
            
            # Target (i) and source (j) nodes of edges
            data['idnb_i'] = edgeid_to_target
            data['idnb_j'] = edgeid_to_source
            
            # Indices of triplets k->j->i
            ntriplets = adj_matrix[edgeid_to_source].sum(1).A1
            id3ynb_i = np.repeat(edgeid_to_target, ntriplets)
            id3ynb_j = np.repeat(edgeid_to_source, ntriplets)
            id3ynb_k = adj_matrix[edgeid_to_source].nonzero()[1]
            
            
            # Indices of triplets that are not i->j->i
            id3_y_to_d, = (id3ynb_i != id3ynb_k).nonzero()
            data['id3dnb_i'] = id3ynb_i[id3_y_to_d]
            data['id3dnb_j'] = id3ynb_j[id3_y_to_d]
            data['id3dnb_k'] = id3ynb_k[id3_y_to_d]
            
            # Edge indices for interactions
            # j->i => k->j
            data['id_expand_kj'] = atomids_to_edgeid[edgeid_to_source, :].data[id3_y_to_d]
            # j->i => k->j => j->i
            data['id_reduce_ji'] = atomids_to_edgeid[edgeid_to_source, :].tocoo().row[id3_y_to_d]

            
            data['Z'] = np.pad(data['Z'], ((0, Z_max-data['Z'].shape[0])), 'constant', constant_values=0)
            data['R'] = np.pad(data['R'], ((0, Z_max-data['R'].shape[0]), (0, 0)), 'constant', constant_values=0.0)
            data['batch_seg'] = np.pad(data['batch_seg'], ((0, Z_max-data['batch_seg'].shape[0])), 'constant', constant_values=batch_size)
            data['idnb_i'] = np.pad(data['idnb_i'], ((0, idnb_i_max-data['idnb_i'].shape[0])), 'constant', constant_values=Z_max-1)
            data['idnb_j'] = np.pad(data['idnb_j'], ((0, idnb_i_max-data['idnb_j'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id_expand_kj'] = np.pad(data['id_expand_kj'], ((0, id3dnb_i_max-data['id_expand_kj'].shape[0])), 'constant', constant_values=idnb_i_max-1)
            data['id_reduce_ji'] = np.pad(data['id_reduce_ji'], ((0, id3dnb_i_max-data['id_reduce_ji'].shape[0])), 'constant', constant_values=idnb_i_max-1)
            data['id3dnb_i'] = np.pad(data['id3dnb_i'], ((0, id3dnb_i_max-data['id3dnb_i'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id3dnb_j'] = np.pad(data['id3dnb_j'], ((0, id3dnb_i_max-data['id3dnb_j'].shape[0])), 'constant', constant_values=Z_max-1)
            data['id3dnb_k'] = np.pad(data['id3dnb_k'], ((0, id3dnb_i_max-data['id3dnb_k'].shape[0])), 'constant', constant_values=Z_max-1)
            
            preds = net(Tensor(data['Z']), Tensor(data['R']), Tensor(data['batch_seg']), Tensor(data['idnb_i']), Tensor(data['idnb_j']), Tensor(data['id_expand_kj'], dtype=mstype.int32), Tensor(data['id_reduce_ji']), Tensor(data['id3dnb_i']), Tensor(data['id3dnb_j']), Tensor(data['id3dnb_k']))
            preds = preds[0:batch_size]
            loss = ops.ReduceMean()(ops.abs(Tensor(data['targets']) - preds))
            total_loss_test = total_loss_test + loss / ((len(all_idx) - 120000) // batch_size)
        print('============loss_test==========')
        print(total_loss_test)
    exit(0)
    
    


if __name__ == '__main__':
    main()
