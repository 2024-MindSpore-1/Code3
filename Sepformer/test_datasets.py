import os
import random
import logging
import sys
from types import SimpleNamespace
from enum import Enum, auto
from create_directory import create_experiment_directory
from hyperpyyaml import load_hyperpyyaml
from dataprocess import dataio, dataset, data_pipeline
from myparser import parse_arguments
from mindspore.dataset import GeneratorDataset, RandomSampler

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore import context
from mindspore.common import set_seed
import mindspore.common.dtype as mstype
from mindspore.train.model import Model, ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from mindspore.train.loss_scale_manager import FixedLossScaleManager


from mindspore.ops import stop_gradient

from src.losses import get_si_snr_with_pitwrapper
from src.dual_path import Encoder, Dual_Path_Model, Decoder, SBTransformerBlock
from train_mindspore import SepFormer, NetWithLoss



if __name__ == '__main__':
    target = "Ascend"
    device_id = 7

    # init context
    # context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
    context.set_context(device_id=device_id)
    # if config.run_distribute:
    #     if target == "Ascend":
    #         device_id = int(os.getenv('DEVICE_ID'))
    #         context.set_context(device_id=device_id)
    #         group_size = 8
    #         context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
    #                                           gradients_mean=False)

    hparams_file, run_opts, overrides = parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    # Logger info
    logger = logging.getLogger(__name__)

    create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    # Data preparation
    from prepare_data import prepare_librimix

    prepare_librimix(
        hparams["data_folder"],
        hparams["save_folder"],
        hparams["num_spks"],
        hparams["skip_prep"],
        hparams["use_wham_noise"],
        hparams["sample_rate"]
    )
    # Create dataset objects
    train_data, valid_data, test_data, label = dataset.dataio_prep(hparams)

    # Create dataloader
    # train_dataset = GeneratorDataset(train_data, label)
    # sample = RandomSampler()
    # train_dataset.add_sampler(sample)
    # train_dataset = train_dataset.batch(1, True)
    # iterator = train_dataset.create_tuple_iterator()

    from src.datasets import create_dataset
    train_dataset = create_dataset(train_data, label, batch_size=1)
    # network = SepFormer(hparams)
    # net_with_loss = NetWithLoss(network, hparams)
    # network.set_train(True)

    # label = ["id", "mix_sig", "s1_sig", "s2_sig"]
    max_mix_sig = 0
    max_id = -1
    # max_s1_sig = 0
    # max_s2_sig = 0
    for _, data in enumerate(train_dataset.create_dict_iterator()):
        # print(_)
        id = data["id"]
        mix_sig = data["mix_sig"]
        s1_sig = data["s1_sig"]
        s2_sig = data["s2_sig"]
        print(id ," mix_sig.shape = ", mix_sig.shape[1])
        # print("s1_sig.shape = ", s1_sig.shape)
        # print("s2_sig.shape = ", s2_sig.shape)
        if mix_sig.shape[1] > max_mix_sig:
            max_id = id
            max_mix_sig = mix_sig.shape[1]
    print("max_id = ", max_id)
    print("max_mix_sig = ", max_mix_sig)
        # loss = net_with_loss(id, mix_sig, s1_sig, s2_sig)
        # print("loss: ", loss)