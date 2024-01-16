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

from src.data.augment.auto_augment import pil_interp, rand_augment_transform
from src.data.augment.mixup import Mixup
from src.data.augment.random_erasing import RandomErasing
from src.data.augment.transforms import RandomResizedCropAndInterpolation, Resize
from .data_utils.moxing_adapter import sync_data


class ImageNet:
    """ImageNet Define"""

    def __init__(self, args, training=True):
        if args.run_modelarts:
            print('Syncing data.')
            data_url = args.data_url
            local_data_path = '/home/work/user-job-dir/inputs/data'
            os.makedirs(local_data_path, exist_ok=True)
            sync_data(data_url, local_data_path, threads=256)
            print(f"local_data_path:{os.listdir(local_data_path)}")
            if "imagenet" in os.listdir(local_data_path):
                local_data_path = os.path.join(local_data_path, "imagenet")
            elif "imagenet.zip" in os.listdir(local_data_path):
                zip_file = zipfile.ZipFile(os.path.join(local_data_path, "imagenet.zip"))
                for file in zip_file.namelist():
                    zip_file.extract(file, local_data_path)
                local_data_path = os.path.join(local_data_path, "imagenet")
            else:
                exit(1)

            train_dir = os.path.join(local_data_path, "train")
            val_ir = os.path.join(local_data_path, "val")
            self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args)
        else:
            train_dir = os.path.join(args.data_url, "train")
            val_ir = os.path.join(args.data_url, "val")
            if training:
                self.train_dataset = create_dataset_imagenet(train_dir, training=True, args=args)
            self.val_dataset = create_dataset_imagenet(val_ir, training=False, args=args)


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
