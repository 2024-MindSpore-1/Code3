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
from src.data.augment.auto_augment import pil_interp, rand_augment_transform
from src.data.augment.mixup import Mixup
from src.data.augment.random_erasing import RandomErasing
from src.data.augment.transforms import RandomResizedCropAndInterpolation, Resize
from .data_utils.moxing_adapter import sync_data


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

class bindingDB:
    """bindingDB Define"""

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
            

            if "BindingDB_train" in os.listdir(local_data_path):
                print('=============4=============')
                local_data_path_train = os.path.join(local_data_path, "BindingDB_train")
            elif "BindingDB_train.zip" in os.listdir(local_data_path):
                print('=============5=============')
                zip_file = zipfile.ZipFile(os.path.join(local_data_path, "BindingDB_train.zip"))
                for file in zip_file.namelist():
                    zip_file.extract(file, local_data_path)
                print('=============6=============')
                local_data_path_train = os.path.join(local_data_path, "BindingDB_train")
            else:
                print('=============7=============')
                exit(1)
            if "BindingDB_dev" in os.listdir(local_data_path):
                print('=============8=============')
                local_data_path_dev = os.path.join(local_data_path, "BindingDB_dev")
            elif "BindingDB_dev.zip" in os.listdir(local_data_path):
                print('=============9=============')
                zip_file = zipfile.ZipFile(os.path.join(local_data_path, "BindingDB_dev.zip"))
                for file in zip_file.namelist():
                    zip_file.extract(file, local_data_path)
                print('=============10=============')    
                local_data_path_dev = os.path.join(local_data_path, "BindingDB_dev")
            else:
                print('=============11=============')
                exit(1)
            if "BindingDB_test" in os.listdir(local_data_path):
                print('=============12=============')
                local_data_path_test = os.path.join(local_data_path, "BindingDB_test")
            elif "BindingDB_test.zip" in os.listdir(local_data_path):
                print('=============13=============')
                zip_file = zipfile.ZipFile(os.path.join(local_data_path, "BindingDB_test.zip"))
                for file in zip_file.namelist():
                    zip_file.extract(file, local_data_path)
                print('=============14=============')
                local_data_path_test = os.path.join(local_data_path, "BindingDB_test")
            else:
                print('=============15=============')
                exit(1)

            print('======local_data_path_train&dev&test_start==================================')
            print(local_data_path_train)
            print(local_data_path_dev)
            print(local_data_path_test)
            print('======local_data_path_train&dev&test_end==================================')
            
            '''
            compounds_train = np.load(os.path.join(local_data_path_train, "word2vec_30/compounds.npy"), allow_pickle=True)
            adjacencies_train = np.load(os.path.join(local_data_path_train, "word2vec_30/adjacencies.npy"), allow_pickle=True)
            proteins_train = np.load(os.path.join(local_data_path_train, "word2vec_30/proteins.npy"), allow_pickle=True)
            interactions_train = np.load(os.path.join(local_data_path_train, "word2vec_30/interactions.npy"), allow_pickle=True)
            train_data = list(zip(compounds_train, adjacencies_train, proteins_train))
            #np.random.seed(1234)
            #np.random.shuffle(dataset_train)
            dataset_train = ds.NumpySlicesDataset((train_data, interactions_train), ["data", "label"])
            dataset_train = dataset_train.map(operations=transforms.TypeCast(mindspore.int32), input_columns="label")
            dataset_train = dataset_train.map(operations=transforms.TypeCast(mindspore.float32), input_columns="data")
            dataset_train = dataset_train.batch(batch_size=2)
            # shuffle?
            self.train_dataset = dataset_train
            exit(0)
            '''
            #compounds_test = [np.array(d.tolist(), dtype=np.uint8) for d in np.load(os.path.join(local_data_path_test, "word2vec_30/compounds.npy"), allow_pickle=True)]
            '''
            for d in np.load(os.path.join(local_data_path_test, "word2vec_30/compounds.npy"), allow_pickle=True):
                print(d)
                print(d.shape)
                print(d.dtype)
                exit(0)
            '''    
            #hh = np.load(os.path.join(local_data_path_test, "word2vec_30/compounds.npy"), allow_pickle=True)
            #compounds_test = []
            #for d in hh:
            #    mm = d.tolist()
            #    print('=======mm=========')
            #    print(mm)
            #    cc = np.array(mm, dtype=np.uint8)
            #    compounds_test.append(cc)
            compounds_train = [d for d in np.load(os.path.join(local_data_path_train, "word2vec_30/compounds.npy"), allow_pickle=True)]
            adjacencies_train = [d for d in np.load(os.path.join(local_data_path_train, "word2vec_30/adjacencies.npy"), allow_pickle=True)]
            proteins_train = [d for d in np.load(os.path.join(local_data_path_train, "word2vec_30/proteins.npy"), allow_pickle=True)]
            interactions_train = [d for d in np.load(os.path.join(local_data_path_train, "word2vec_30/interactions.npy"), allow_pickle=True)]
            train_data = list(zip(compounds_train, adjacencies_train, proteins_train, interactions_train))
            
            compounds_test = [d for d in np.load(os.path.join(local_data_path_test, "word2vec_30/compounds.npy"), allow_pickle=True)]
            adjacencies_test = [d for d in np.load(os.path.join(local_data_path_test, "word2vec_30/adjacencies.npy"), allow_pickle=True)]
            proteins_test = [d for d in np.load(os.path.join(local_data_path_test, "word2vec_30/proteins.npy"), allow_pickle=True)]
            interactions_test = [d for d in np.load(os.path.join(local_data_path_test, "word2vec_30/interactions.npy"), allow_pickle=True)]
            test_data = list(zip(compounds_test, adjacencies_test, proteins_test, interactions_test))
            
            train_data = shuffle_dataset(train_data, 1234)
            self.train_data = train_data
            self.test_data = test_data
            #dataset_test = ds.NumpySlicesDataset((test_data, interactions_test), ["data", "label"])
            #data1 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8) 
            #data2 = np.array(np.random.sample(size=(300, 300, 3)) * 255, dtype=np.uint8) 
            #data1 = compounds_test[0]
            #data2 = compounds_test[1]
            #data3 = np.array(np.random.sample(size=(28, 34)) * 255, dtype=np.uint8)
            #data4 = np.array(np.random.sample(size=(29, 34)) * 255, dtype=np.uint8)
           
            #print('=====data1======')
            #print(data1)
            #print('=====data3======')
            #print(data3)
            #print('=====data1.shape======')
            #print(data1.shape)
            #print('=====data3.shape======')
            #print(data3.shape)
            #print('=====data1.dtype======')
            #print(data1.dtype)
            #print('=====data3.dtype======')
            #print(data3.dtype)
            #data = ([data3, data4], [3, 4])
            #dataset_test = ds.NumpySlicesDataset(data=data, column_names=["compounds", "adjacencies", "proteins", "label"])
            #dataset_test = ds.NumpySlicesDataset(data=data, column_names=["data", "label"])
            #print(compounds_test[0])
            #print(compounds_test[0].shape)
            #dataset_test = ds.NumpySlicesDataset((compounds_test, adjacencies_test, proteins_test, interactions_test), ["compounds", "adjacencies", "proteins", "label"])
            #dataset_test = dataset_test.map(operations=C.TypeCast(mindspore.int32), input_columns="label")
            #dataset_test = dataset_test.map(operations=C.TypeCast(mindspore.float32), input_columns="compounds")
            #dataset_test = dataset_test.map(operations=C.TypeCast(mindspore.float32), input_columns="adjacencies")
            #dataset_test = dataset_test.map(operations=C.TypeCast(mindspore.float32), input_columns="proteins")
            #dataset_test = dataset_test.batch(batch_size=2)
            #self.test_dataset = dataset_test
            #self.train_dataset = dataset_test
            #exit(0)
        
        else:
            print('Syncing data.')
            
            local_data_path = '/home/ma-user/work/inputs/data'
            #local_data_path = '/home/work/user-job-dir/inputs/data'
            os.makedirs(local_data_path, exist_ok=True)
            print('=============1=============')
            print(args.multi_data_url)
            multi_data_url_list = args.multi_data_url[1:][:-1].replace('{', '').replace('}', '').split(',')

            url = 's3:///urchincache/attachment/d/d/ddf3cf33-fc14-49f7-8958-56eefc437aeb'
            sync_data(url, local_data_path, threads=256)
            url = 's3:///urchincache/attachment/5/8/5865551c-6946-4498-989a-d7cd7764c049'
            sync_data(url, local_data_path, threads=256)
            if "BindingDB_train" in os.listdir(local_data_path):
                print('=============4=============')
                local_data_path_train = os.path.join(local_data_path, "BindingDB_train")
            elif "BindingDB_train.zip" in os.listdir(local_data_path):
                print('=============5=============')
                zip_file = zipfile.ZipFile(os.path.join(local_data_path, "BindingDB_train.zip"))
                for file in zip_file.namelist():
                    zip_file.extract(file, local_data_path)
                print('=============6=============')
                local_data_path_train = os.path.join(local_data_path, "BindingDB_train")
            else:
                print('=============7=============')
                exit(1)
            if "BindingDB_test" in os.listdir(local_data_path):
                print('=============12=============')
                local_data_path_test = os.path.join(local_data_path, "BindingDB_test")
            elif "BindingDB_test.zip" in os.listdir(local_data_path):
                print('=============13=============')
                zip_file = zipfile.ZipFile(os.path.join(local_data_path, "BindingDB_test.zip"))
                for file in zip_file.namelist():
                    zip_file.extract(file, local_data_path)
                print('=============14=============')
                local_data_path_test = os.path.join(local_data_path, "BindingDB_test")
            else:
                print('=============15=============')
                exit(1)

            print('======local_data_path_train&dev&test_start==================================')
            print(local_data_path_test)
            print('======local_data_path_train&dev&test_end==================================')
            
            compounds_train = [d for d in np.load(os.path.join(local_data_path_train, "word2vec_30/compounds.npy"), allow_pickle=True)]
            adjacencies_train = [d for d in np.load(os.path.join(local_data_path_train, "word2vec_30/adjacencies.npy"), allow_pickle=True)]
            proteins_train = [d for d in np.load(os.path.join(local_data_path_train, "word2vec_30/proteins.npy"), allow_pickle=True)]
            interactions_train = [d for d in np.load(os.path.join(local_data_path_train, "word2vec_30/interactions.npy"), allow_pickle=True)]
            train_data = list(zip(compounds_train, adjacencies_train, proteins_train, interactions_train))
            


            compounds_test = [d for d in np.load(os.path.join(local_data_path_test, "word2vec_30/compounds.npy"), allow_pickle=True)]
            adjacencies_test = [d for d in np.load(os.path.join(local_data_path_test, "word2vec_30/adjacencies.npy"), allow_pickle=True)]
            proteins_test = [d for d in np.load(os.path.join(local_data_path_test, "word2vec_30/proteins.npy"), allow_pickle=True)]
            interactions_test = [d for d in np.load(os.path.join(local_data_path_test, "word2vec_30/interactions.npy"), allow_pickle=True)]
            test_data = list(zip(compounds_test, adjacencies_test, proteins_test, interactions_test))
            
            train_data = shuffle_dataset(train_data, 1234)
            self.train_data = train_data
            self.test_data = test_data
            

           

        


def create_dataset_imagenet(dataset_dir, args, repeat_num=1, training=True):
    """
    create a train or eval imagenet2012 dataset for TNT

    Args:
        dataset_dir(string): the path of dataset.
        do_train(bool): whether dataset is used for train or eval.
        repeat_num(int): the repeat times of dataset. Default: 1

    Returns:
        dataset
    """

    device_num, rank_id = _get_rank_info()
    shuffle = training
    if device_num == 1 or not training:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers,
                                         shuffle=shuffle)
    else:
        data_set = ds.ImageFolderDataset(dataset_dir, num_parallel_workers=args.num_parallel_workers, shuffle=shuffle,
                                         num_shards=device_num, shard_id=rank_id)

    image_size = args.image_size

    # define map operations
    # BICUBIC: 3
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if training:
        aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        interpolation = args.interpolation
        auto_augment = args.auto_augment
        if interpolation != "random":
            aa_params["interpolation"] = pil_interp(interpolation)
        assert auto_augment.startswith('rand')
        transform_img = [
            vision.Decode(),
            py_vision.ToPIL(),
            RandomResizedCropAndInterpolation(size=args.image_size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.),
                                              interpolation=interpolation),
            py_vision.RandomHorizontalFlip(prob=0.5),
        ]
        transform_img += [rand_augment_transform(auto_augment, aa_params)]
        transform_img += [
            py_vision.ToTensor(),
            py_vision.Normalize(mean=mean, std=std)]
        if args.re_prob > 0.:
            transform_img += [RandomErasing(args.re_prob, mode=args.re_mode, max_count=args.re_count)]
    else:
        # test transform complete
        transform_img = [
            vision.Decode(),
            py_vision.ToPIL(),
            Resize(int(args.image_size / args.crop_pct), interpolation=args.interpolation),
            py_vision.CenterCrop(image_size),
            py_vision.ToTensor(),
            py_vision.Normalize(mean=mean, std=std)
        ]
    transform_label = C.TypeCast(mstype.int32)

    data_set = data_set.map(input_columns="image", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_img)
    data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                            operations=transform_label)
    if (args.mix_up > 0. or args.cutmix > 0.) and not training:
        # if use mixup and not training(False), one hot val data label
        one_hot = C.OneHot(num_classes=args.num_classes)
        data_set = data_set.map(input_columns="label", num_parallel_workers=args.num_parallel_workers,
                                operations=one_hot)
    # apply batch operations
    data_set = data_set.batch(args.batch_size, drop_remainder=True,
                              num_parallel_workers=args.num_parallel_workers)

    if (args.mix_up > 0. or args.cutmix > 0.) and training:
        mixup_fn = Mixup(
            mixup_alpha=args.mix_up, cutmix_alpha=args.cutmix, cutmix_minmax=None,
            prob=args.mixup_prob, switch_prob=args.switch_prob, mode=args.mixup_mode,
            label_smoothing=args.label_smoothing, num_classes=args.num_classes)

        data_set = data_set.map(operations=mixup_fn, input_columns=["image", "label"],
                                num_parallel_workers=args.num_parallel_workers)

    # apply dataset repeat operation
    data_set = data_set.repeat(repeat_num)

    return data_set


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
