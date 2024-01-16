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
"""
Data operations, will be used in train.py and eval.py
"""
import os
import zipfile
import time

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as vision
import mindspore.dataset.vision.py_transforms as py_vision
import numpy as np
import mindspore 
import mindspore.dataset.transforms as transforms

from mindspore import Tensor
from mindspore import ops
from .data_utils.moxing_adapter import sync_data

target = 'U0'

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

class qm9:
    """qm9 Define"""

    def __init__(self, args, training=True):
        if args.run_modelarts:
            print('Syncing data.')
            

            local_data_path = '/home/work/user-job-dir/inputs/data'
            os.makedirs(local_data_path, exist_ok=True)
            print('=============1=============')
            print(args.multi_data_url)
            multi_data_url_list = args.multi_data_url[1:][:-1].replace('{', '').replace('}', '').split(',')
            
            for list_index in range(len(multi_data_url_list)):
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                if list_index % 2 == 0:  
                    url = multi_data_url_list[list_index][15: -1]
                    print(url)
                    sync_data(url, local_data_path, threads=256)
            print('=============2=============')
            print(f"local_data_path:{os.listdir(local_data_path)}")
            print('=============3=============')
            

            if 'qm9_eV.npz' in os.listdir(local_data_path):
                print('=============4=============')
                local_data_path = os.path.join(local_data_path, "qm9_eV.npz")
            elif 'qm9.zip' in os.listdir(local_data_path):
                print('=============5=============')
                exit(1)
            else:
                print('=============7=============')
                exit(1)


            print('======local_data_path==================================')
            print(local_data_path)
            print('======local_data_path==================================')
            
            data_dict = np.load(local_data_path, allow_pickle=True)
            self.id = data_dict['id'].astype(np.int32)
            self.N = data_dict['N'].astype(np.int32)
            self.Z = data_dict['Z'].astype(np.int32)
            self.R = data_dict['R'].astype(np.float64)
            self.targets = np.stack([data_dict[target]], axis=1).astype(np.float32)
            self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
        
        else:
            print('Syncing data.')
            
            local_data_path = '/home/ma-user/work/inputs/data'
            #local_data_path = '/home/work/user-job-dir/inputs/data'
            os.makedirs(local_data_path, exist_ok=True)
            print('=============1=============')
            print(args.multi_data_url)
            multi_data_url_list = args.multi_data_url[1:][:-1].replace('{', '').replace('}', '').split(',')
            
            print('=========QAQ=======')
            print(os.listdir(local_data_path))
            if 'qm9_eV.npz' in os.listdir(local_data_path):
                print('=============4=============')
                local_data_path = os.path.join(local_data_path, "qm9_eV.npz")
            elif 'qm9.zip' in os.listdir(local_data_path):
                print('=============5=============')
                exit(1)
            else:
                print('=============7=============')
                exit(1)


            print('======local_data_path==================================')
            print(local_data_path)
            print('======local_data_path==================================')
            
            data_dict = np.load(local_data_path, allow_pickle=True)
            self.id = data_dict['id'].astype(np.int32)
            self.N = data_dict['N'].astype(np.int32)
            self.Z = data_dict['Z'].astype(np.int32)
            self.R = data_dict['R'].astype(np.float64)
            self.targets = np.stack([data_dict[target]], axis=1).astype(np.float32)
            self.N_cumsum = np.concatenate([[0], np.cumsum(self.N)])
            #data = (Z.tolist(), R.tolist())
            #dataset = ds.NumpySlicesDataset(data=data, column_names=["Z", "R"])
            

def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
