import contextlib
import logging
from . import data_pipeline, dataio
from .data_pipeline import DataPipeline
from .dataio import load_data_csv

logger = logging.getLogger(__name__)


class DynamicItemDataset():
    """Dataset that reads, wrangles, and produces dicts.
    Arguments
    ---------
    data : dict
        Dictionary containing single data points (e.g. utterances).
    dynamic_items : list, optional
        Configuration for the dynamic items produced when fetching an example.
        List of DynamicItems or dicts with the format::
            func: <callable> # To be called
            takes: <list> # key or list of keys of args this takes
            provides: key # key or list of keys that this provides
    output_keys : dict, list, optional
        List of keys (either directly available in data or dynamic items)
        to include in the output dict when data points are fetched.

        If a dict is given; it is used to map internal keys to output keys.
        From the output_keys dict key:value pairs the key appears outside,
        and value is the internal key.
    """

    def __init__(
            self, data, dynamic_items=[], output_keys=[],
    ):
        self.data = data
        self.data_ids = list(self.data.keys())
        static_keys = list(self.data[self.data_ids[0]].keys())
        if "id" in static_keys:
            raise ValueError("The key 'id' is reserved for the data point id.")
        else:
            static_keys.append("id")
        self.pipeline = DataPipeline(static_keys, dynamic_items)
        self.set_output_keys(output_keys)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        data_id = self.data_ids[index]
        data_point = self.data[data_id]
        return self.pipeline.compute_outputs({"id": data_id, **data_point})

    def add_dynamic_item(self, func, takes=None, provides=None):
        """Makes a new dynamic item available on the dataset.

        Two calling conventions. For DynamicItem objects, just use:
        add_dynamic_item(dynamic_item).
        But otherwise, should use:
        add_dynamic_item(func, takes, provides).

        See `speechbrain.utils.data_pipeline`.

        Arguments
        ---------
        func : callable, DynamicItem
            If a DynamicItem is given, adds that directly. Otherwise a
            DynamicItem is created, and this specifies the callable to use. If
            a generator function is given, then create a GeneratorDynamicItem.
            Otherwise creates a normal DynamicItem.
        takes : list, str
            List of keys. When func is called, each key is resolved to
            either an entry in the data or the output of another dynamic_item.
            The func is then called with these as positional arguments,
            in the same order as specified here.
            A single arg can be given directly.
        provides : str
            Unique key or keys that this provides.
        """
        self.pipeline.add_dynamic_item(func, takes, provides)

    def set_output_keys(self, keys):
        """Use this to change the output keys.

        These are the keys that are actually evaluated when a data point
        is fetched from the dataset.

        Arguments
        ---------
        keys : dict, list
            List of keys (str) to produce in output.

            If a dict is given; it is used to map internal keys to output keys.
            From the output_keys dict key:value pairs the key appears outside,
            and value is the internal key.
        """
        self.pipeline.set_output_keys(keys)

    @contextlib.contextmanager
    def output_keys_as(self, keys):
        """Context manager to temporarily set output keys.
        """
        saved_output = self.pipeline.output_mapping
        self.pipeline.set_output_keys(keys)
        yield self
        self.pipeline.set_output_keys(saved_output)

    def _filtered_sorted_ids(
            self,
            key_min_value={},
            key_max_value={},
            key_test={},
            sort_key=None,
            reverse=False,
            select_n=None,
    ):
        """Returns a list of data ids, fulfilling the sorting and filtering."""

        def combined_filter(computed):
            for key, limit in key_min_value.items():
                # NOTE: docstring promises >= so using that.
                # Mathematically could also use < for nicer syntax, but
                # maybe with some super special weird edge case some one can
                # depend on the >= operator
                if computed[key] >= limit:
                    continue
                return False
            for key, limit in key_max_value.items():
                if computed[key] <= limit:
                    continue
                return False
            for key, func in key_test.items():
                if bool(func(computed[key])):
                    continue
                return False
            return True

        temp_keys = (
                set(key_min_value.keys())
                | set(key_max_value.keys())
                | set(key_test.keys())
                | set([] if sort_key is None else [sort_key])
        )
        filtered_ids = []
        with self.output_keys_as(temp_keys):
            for i, data_id in enumerate(self.data_ids):
                if select_n is not None and len(filtered_ids) == select_n:
                    break
                data_point = self.data[data_id]
                data_point["id"] = data_id
                computed = self.pipeline.compute_outputs(data_point)
                if combined_filter(computed):
                    if sort_key is not None:
                        # Add (main sorting index, current index, data_id)
                        # So that we maintain current sorting and don't compare
                        # data_id values ever.
                        filtered_ids.append((computed[sort_key], i, data_id))
                    else:
                        filtered_ids.append(data_id)
        if sort_key is not None:
            filtered_sorted_ids = [
                tup[2] for tup in sorted(filtered_ids, reverse=reverse)
            ]
        else:
            filtered_sorted_ids = filtered_ids
        return filtered_sorted_ids

    @classmethod
    def from_csv(
            cls, csv_path, replacements={}, dynamic_items=[], output_keys=[]
    ):
        """Load a data prep CSV file and create a Dataset based on it."""
        data = load_data_csv(csv_path, replacements)
        return cls(data, dynamic_items, output_keys)


def add_dynamic_item(datasets, func, takes=None, provides=None):
    """Helper for adding the same item to multiple datasets."""
    for dataset in datasets:
        dataset.add_dynamic_item(func, takes, provides)


def set_output_keys(datasets, output_keys):
    """Helper for setting the same item to multiple datasets."""
    for dataset in datasets:
        dataset.set_output_keys(output_keys)


def dataio_prep(hparams):
    """Creates data processing pipeline"""

    # 1. Define datasets
    train_data = DynamicItemDataset.from_csv(
        csv_path=hparams["train_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    valid_data = DynamicItemDataset.from_csv(
        csv_path=hparams["valid_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    test_data = DynamicItemDataset.from_csv(
        csv_path=hparams["test_data"],
        replacements={"data_root": hparams["data_folder"]},
    )

    datasets = [train_data, valid_data, test_data]

    # 2. Provide audio pipelines

    @data_pipeline.takes("mix_wav")
    @data_pipeline.provides("mix_sig")
    def audio_pipeline_mix(mix_wav):
        mix_sig = dataio.read_audio(mix_wav)
        return mix_sig

    @data_pipeline.takes("s1_wav")
    @data_pipeline.provides("s1_sig")
    def audio_pipeline_s1(s1_wav):
        s1_sig = dataio.read_audio(s1_wav)
        return s1_sig

    @data_pipeline.takes("s2_wav")
    @data_pipeline.provides("s2_sig")
    def audio_pipeline_s2(s2_wav):
        s2_sig = dataio.read_audio(s2_wav)
        return s2_sig

    if hparams["num_spks"] == 3:
        @data_pipeline.takes("s3_wav")
        @data_pipeline.provides("s3_sig")
        def audio_pipeline_s3(s3_wav):
            s3_sig = dataio.read_audio(s3_wav)
            return s3_sig

    if hparams["use_wham_noise"]:
        @data_pipeline.takes("noise_wav")
        @data_pipeline.provides("noise_sig")
        def audio_pipeline_noise(noise_wav):
            noise_sig = dataio.read_audio(noise_wav)
            return noise_sig

    add_dynamic_item(datasets, audio_pipeline_mix)
    add_dynamic_item(datasets, audio_pipeline_s1)
    add_dynamic_item(datasets, audio_pipeline_s2)
    if hparams["num_spks"] == 3:
        add_dynamic_item(datasets, audio_pipeline_s3)

    if hparams["use_wham_noise"]:
        print("Using the WHAM! noise in the data pipeline")
        add_dynamic_item(datasets, audio_pipeline_noise)

    if (hparams["num_spks"] == 2) and hparams["use_wham_noise"]:
        label = ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
    elif (hparams["num_spks"] == 3) and hparams["use_wham_noise"]:
        label = ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig", "noise_sig"]
    elif (hparams["num_spks"] == 2) and not hparams["use_wham_noise"]:
        label = ["id", "mix_sig", "s1_sig", "s2_sig"]
    else:
        label = ["id", "mix_sig", "s1_sig", "s2_sig", "s3_sig"]
    set_output_keys(datasets, label)
    return train_data, valid_data, test_data, label


from mindspore.dataset import GeneratorDataset, RandomSampler
import os
import mindspore.dataset.transforms.c_transforms as C2
import mindspore.common.dtype as mstype

# def create_dataset(train_data, batch_size=1, target="Ascend"):
#     """
#     create a train dataset
#     """
#     if target == "Ascend":
#         rank_size, rank_id = _get_rank_info()
#
#     train_dataset = GeneratorDataset(train_data, column_names=label,
#                                      num_parallel_workers=8,
#                                      shuffle=True,
#                                      num_shards=rank_size,
#                                      shard_id=rank_id)
#
#     sample = RandomSampler()
#     train_dataset.add_sampler(sample)
#     train_dataset = train_dataset.batch(batch_size=batch_size, drop_remainder=True, num_parallel_workers=8)
#     # iterator = train_dataset.create_tuple_iterator()
#     return train_dataset
#
#
# def _get_rank_info():
#     """
#     get rank size and rank id
#     """
#     rank_size = int(os.environ.get("RANK_SIZE", 1))
#
#     if rank_size > 1:
#         rank_size = int(os.environ.get("RANK_SIZE"))
#         rank_id = int(os.environ.get("RANK_ID"))
#     else:
#         rank_size = 1
#         rank_id = 0
#
#     return rank_size, rank_id


def create_dataset(hparams, batch_size=1, target="Ascend"):
    """
    create a train dataset
    """
    # if target == "Ascend":
    rank_size, rank_id = _get_rank_info()

    train_data, valid_data, test_data, label = dataio_prep(hparams)

    train_dataset = GeneratorDataset(train_data, column_names=label,
                                     num_parallel_workers=8,
                                     shuffle=True,
                                     num_shards=rank_size,
                                     shard_id=rank_id)

    # sample = RandomSampler()
    # train_dataset.add_sampler(sample)
    # ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
    type_cast_float32_op = C2.TypeCast(mstype.float32)
    train_dataset = train_dataset.map(operations=type_cast_float32_op, input_columns="mix_sig",
                                      num_parallel_workers=8)
    train_dataset = train_dataset.map(operations=type_cast_float32_op, input_columns="s1_sig",
                                      num_parallel_workers=8)
    train_dataset = train_dataset.map(operations=type_cast_float32_op, input_columns="s2_sig",
                                      num_parallel_workers=8)
    type_cast_int32_op = C2.TypeCast(mstype.int32)
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
