import os

# import mindspore.dataset.transforms.c_transforms as C2
from mindspore.dataset import transforms
import mindspore.common.dtype as mstype
from mindspore.dataset import GeneratorDataset, RandomSampler

def create_dataset(train_data, label, batch_size=1, target="Ascend"):
    """
    create a train dataset
    """
    # if target == "Ascend":
    rank_size, rank_id = _get_rank_info()

    train_dataset = GeneratorDataset(train_data, column_names=label,
                                     num_parallel_workers=1,
                                     shuffle=True,
                                     num_shards=rank_size,
                                     shard_id=rank_id)

    # sample = RandomSampler()
    # train_dataset.add_sampler(sample)
    # ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
    # type_cast_float32_op = C2.TypeCast(mstype.float32)
    type_cast_float32_op = transforms.TypeCast(mstype.float32)
    train_dataset = train_dataset.map(operations=type_cast_float32_op, input_columns="mix_sig",
                                      num_parallel_workers=8)
    train_dataset = train_dataset.map(operations=type_cast_float32_op, input_columns="s1_sig",
                                      num_parallel_workers=8)
    train_dataset = train_dataset.map(operations=type_cast_float32_op, input_columns="s2_sig",
                                      num_parallel_workers=8)
    # type_cast_int32_op = C2.TypeCast(mstype.int32)
    type_cast_int32_op = transforms.TypeCast(mstype.int32)
    train_dataset = train_dataset.map(operations=type_cast_int32_op, input_columns="id", num_parallel_workers=8)
    train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True, num_parallel_workers=8)
    # iterator = train_dataset.create_tuple_iterator()
    return train_dataset


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        rank_size = int(os.environ.get("RANK_SIZE"))
        rank_id = int(os.environ.get("RANK_ID"))
    else:
        rank_size = 1
        rank_id = 0

    return rank_size, rank_id