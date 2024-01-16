"""
dataset
"""
import os
import math
import numpy as np
import mindspore.dataset as ds

np.random.seed(0)
ds.config.set_seed(0)


class AirfoilDataset:
    """
    airfoil 2D-steady problem based on ViT
    Supported Platforms:
        ``Ascend`` ``GPU``

    """
    def __init__(self, max_value, min_value, data_group_size):
        self.max_value_list = max_value
        self.min_value_list = min_value
        self.data_group_size = data_group_size

    def create_dataset(self, data_path, train_num_list, eval_num_list, mode="train", batch_size=8, train_size=0.8,
                       finetune_size=0.2, shuffle=False, drop_remainder=False):
        """
        creeate dataset
        """
        if mode == 'eval':
            eval_dataset = ds.MindDataset(dataset_files=os.path.join(data_path,'test_dataset.mind'), shuffle=shuffle)
            eval_dataset = eval_dataset.project(["inputs", "labels"])
            data_set_norm_eval = eval_dataset.map(operations=self._process_fn, input_columns=["inputs"])
            print("{} eval dataset size: {}".format(mode, data_set_norm_eval.get_dataset_size()))
            data_set_batch_eval = data_set_norm_eval.batch(batch_size, drop_remainder)
            print("{} eavl batch dataset size: {}".format(mode, data_set_batch_eval.get_dataset_size()))
            return None, data_set_batch_eval


        if mode == 'finetune':
            train_dataset = ds.MindDataset(dataset_files=os.path.join(data_path,'train_dataset.mind'), shuffle=shuffle)
            train_dataset = train_dataset.project(["inputs", "labels"])
            eval_dataset = ds.MindDataset(dataset_files=os.path.join(data_path,'test_dataset.mind'), shuffle=shuffle)
            eval_dataset = eval_dataset.project(["inputs", "labels"])

        elif mode == 'train':
            print(data_path+'/train_dataset.mind')
            train_dataset = ds.MindDataset(dataset_files=os.path.join(data_path,'train_dataset.mind'), shuffle=shuffle)
            train_dataset = train_dataset.project(["inputs", "labels"])
            eval_dataset = ds.MindDataset(dataset_files=os.path.join(data_path,'test_dataset.mind'), shuffle=shuffle)
            eval_dataset = eval_dataset.project(["inputs", "labels"])
        train_dataset = train_dataset.shuffle(batch_size * 4)
        eval_dataset = eval_dataset.shuffle(batch_size * 4)

        data_set_norm_train = train_dataset.map(operations=self._process_fn,
                                                input_columns=["inputs"])

        data_set_norm_eval = eval_dataset.map(operations=self._process_fn,
                                              input_columns=["inputs"])

        print("{} dataset : {}".format(mode, train_num_list))
        print("train dataset size: {}".format(data_set_norm_train.get_dataset_size()))
        print("test dataset size: {}".format(data_set_norm_eval.get_dataset_size()))

        data_set_batch_train = data_set_norm_train.batch(batch_size, drop_remainder)
        data_set_batch_eval = data_set_norm_eval.batch(batch_size, drop_remainder)
        print("train batch : {}".format(data_set_batch_train.get_dataset_size()))
        print("test batch : {}".format(data_set_batch_eval.get_dataset_size()))
        return data_set_batch_train, data_set_batch_eval

    def _process_fn(self, data):
        """
        preprocess data
        """
        _, h, w = data.shape
        xex = np.linspace(0, 5, h)
        f4 = np.array([math.exp(-x0) for x0 in xex])
        f4 = np.repeat(f4[:, np.newaxis], w, axis=1)

        aoa = data[0:1, ...]
        x = data[1:2, ...]
        y = data[2:3, ...]
        f4_x = x * f4
        f4_y = y * f4
        data = np.vstack((aoa, f4_x, f4_y))
        eps = 1e-8

        for i in range(0, data.shape[0]):
            max_value, min_value = self.max_value_list[i], self.min_value_list[i]
            data[i, :, :] = (data[i, :, :] - min_value) / (max_value - min_value + eps)
        return data.astype('float32')

    def _select_shards(self, data_path, num_list, prefix="flowfield"):
        """
        select shards file
        """
        name_list = []
        for num in num_list:
            suffix = "_" + (str(num).rjust(3, '0')) + \
                        "_" + (str(num + self.data_group_size - 1).rjust(3, '0')) + '.mind'
            file_name = os.path.join(data_path, prefix + suffix)
            name_list.append(file_name)
        return name_list