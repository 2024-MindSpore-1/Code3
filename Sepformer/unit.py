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


class Stage(Enum):
    """Simple enum to track stage of experiments."""

    TRAIN = auto()
    VALID = auto()
    TEST = auto()


class add_speech_distortions(nn.Cell):
    def __init__(self, hparams=None):
        super(add_speech_distortions, self).__init__()

        self.concat = ops.Concat(axis=-1)
        self.expand_dims = ops.ExpandDims()
        self.zeros = ops.Zeros()
        self.use_speedperturb = hparams["use_speedperturb"]
        self.use_rand_shift = hparams["use_rand_shift"]
        self.use_wham_noise = hparams["use_wham_noise"]
        self.speedperturb = hparams["speedperturb"]

        # Make hyperparams available with dot notation too
        # if hparams is not None:
        #     self.hparams = SimpleNamespace(**hparams)

    def construct(self, mix, targets, stage, noise=None):
        # mix, mix_lens = mix
        mix_lens = 1
        targets = self.concat((self.expand_dims(targets[0], -1), self.expand_dims(targets[1], -1)))

        # Add speech distortions
        if stage == "TRAIN":
            # with torch.no_grad():
            # use_speedperturb: True
            # use_rand_shift: False
            if self.use_speedperturb or self.use_rand_shift:
                mix, targets = self.add_speed_perturb(targets, mix_lens)

                mix = targets.sum(-1)

                # use_wham_noise: False
                if self.use_wham_noise:
                    # noise = noise.to(self.device)
                    len_noise = noise.shape[1]
                    len_mix = mix.shape[1]
                    # min_len = min(len_noise, len_mix)
                    if len_noise < len_mix:
                        min_len = len_noise
                    else:
                        min_len = len_mix

                    # add the noise
                    mix = mix[:, :min_len] + noise[:, :min_len]

                    # fix the length of targets also
                    targets = targets[:, :min_len, :]
            targets = stop_gradient(targets)
            mix = stop_gradient(mix)
        return mix, targets

    def add_speed_perturb(self, targets, targ_lens):
        """Adds speed perturbation and random_shift to the input signals"""

        min_len = -1
        recombine = False

        # use_speedperturb: True
        if self.use_speedperturb:
            # Performing speed change (independently on each source)
            new_targets = []
            recombine = True
            for i in range(targets.shape[-1]):
                new_target = self.speedperturb(
                    targets[:, :, i], targ_lens
                )
                new_targets.append(new_target)
                if i == 0:
                    min_len = new_target.shape[-1]
                else:
                    if new_target.shape[-1] < min_len:
                        min_len = new_target.shape[-1]

            # Re-combination
            if recombine:
                # use_speedperturb: True
                if self.use_speedperturb:
                    # targets = torch.zeros(
                    #     targets.shape[0],
                    #     min_len,
                    #     targets.shape[-1],
                    #     device=targets.device,
                    #     dtype=torch.float,
                    # )
                    targets = self.zeros(
                        (targets.shape[0], min_len, targets.shape[-1]),
                        mindspore.float32
                    )
                for i, new_target in enumerate(new_targets):
                    targets[:, :, i] = new_targets[i][:, 0:min_len]

        mix = targets.sum(-1)
        return mix, targets


class SepFormer(nn.Cell):
    def __init__(self, hparams):
        super(SepFormer, self).__init__()
        self.expand_dims = ops.ExpandDims()
        self.concat = ops.Concat(axis=-1)
        self.stack = ops.Stack()
        self.Encoder = Encoder(kernel_size=16,
                               out_channels=256)
        self.intra_model = SBTransformerBlock(num_layers=8,
                                              d_model=256,
                                              nhead=8,
                                              d_ffn=1024,
                                              dropout=0,
                                              use_positional_encoding=True,
                                              norm_before=True)
        self.inter_model = SBTransformerBlock(num_layers=8,
                                              d_model=256,
                                              nhead=8,
                                              d_ffn=1024,
                                              dropout=0,
                                              use_positional_encoding=True,
                                              norm_before=True)

        self.MaskNet = Dual_Path_Model(in_channels=256,
                                       out_channels=256,
                                       intra_model=self.intra_model,
                                       inter_model=self.inter_model,
                                       num_spks=2,
                                       num_layers=2,
                                       norm="ln",
                                       K=250,
                                       skip_around_intra=True,
                                       linear_layer_after_inter_intra=False)
        self.Decoder = Decoder(in_channels=256,
                               out_channels=1,
                               kernel_size=16,
                               stride=8,
                               has_bias=False)

        # Make hyperparams available with dot notation too
        # if hparams is not None:
        #     self.hparams = SimpleNamespace(**hparams)
        self.add_speech_distortions = add_speech_distortions(hparams)
        self.mul = ops.Mul()
        self.num_spks = hparams["num_spks"]


    def construct(self, mix, targets, stage, noise=None):
        # # Add speech distortions
        mix, targets = self.add_speech_distortions(mix, targets, stage, noise)

        print("mix.shape = ", mix.shape)
        print("targets.shape =", targets.shape)
        # SepFormer
        mix_w = self.Encoder(mix)
        print("mix_w.shape = ", mix_w.shape)
        est_mask = self.MaskNet(mix_w)
        print("est_mask.shape = ", est_mask.shape)
        mix_w = self.stack([mix_w] * self.num_spks)
        # sep_h = mix_w * est_mask
        sep_h = self.mul(mix_w, est_mask)

        # self.print("shape=", sep_h[0].shape)
        # est_source = self.concat(
        #     [
        #         self.expand_dims(self.Decoder(sep_h[i]), -1)
        #         for i in range(self.hparams.num_spks)
        #     ]
        # )
        # Decoding
        est_source = self.concat(
            (
                self.expand_dims(self.Decoder(sep_h[0]), -1),
                self.expand_dims(self.Decoder(sep_h[1]), -1)
            )
        )

        # T changed after conv1d in encoder, fix it here
        T_origin = mix.shape[1]
        T_est = est_source.shape[1]
        if T_origin > T_est:
            # est_source = F.pad(est_source, (0, 0, 0, T_origin - T_est))
            # pad = nn.Pad(paddings=((0, T_origin - T_est), (0, 0)))
            pad = nn.Pad(paddings=((0, 0), (0, T_origin - T_est), (0, 0)))
            est_source = pad(est_source)
        else:
            est_source = est_source[:, :T_origin, :]

        return est_source, targets


# def compute_objectives(predictions, targets):
#     """Computes the si-snr loss"""
#     return get_si_snr_with_pitwrapper(targets, predictions)


class NetWithLoss(nn.Cell):
    """
    NetWithLoss
    """

    def __init__(self, SepFormer, hparams):
        super(NetWithLoss, self).__init__()
        # self.fasttext = FastText(vocab_size, embedding_dims, num_class)
        # self.loss_func = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        # self.squeeze = P.Squeeze(axis=1)
        self.hparams = hparams
        self.compute_forward = SepFormer
        self.size = ops.Size()
        self.use_wham_noise = hparams["use_wham_noise"]
        self.threshold_byloss = hparams["threshold_byloss"]
        self.threshold = hparams["threshold"]
        self.compute_objectives = get_si_snr_with_pitwrapper()

    def construct(self, id, mix_sig, s1_sig, s2_sig):
        """
        SepFormer network with loss.
        """

        # Unpacking batch list
        # mixture = batch.mix_sig
        # targets = [batch.s1_sig, batch.s2_sig]
        # if self.hparams.use_wham_noise:
        #     noise = batch.noise_sig[0]
        # else:
        #     noise = None

        # ["id", "mix_sig", "s1_sig", "s2_sig", "noise_sig"]
        # mixture = batch[1]
        # targets = [batch[2], batch[3]]
        # if self.hparams.use_wham_noise:
        #     noise = batch[4]
        # else:
        #     noise = None

        mixture = mix_sig
        targets = [s1_sig, s2_sig]
        if self.use_wham_noise:
            noise = None
        else:
            noise = None

        predictions, targets = self.compute_forward(
            mixture, targets, "TRAIN", noise
        )
        print("predictions.shape = ", predictions.shape)
        print("targets.shape = ", targets.shape)
       
        loss = self.compute_objectives(predictions, targets)
        print("loss.shape = ", loss.shape)
        print("loss = ", loss)

        # 源代码有
        # # hard threshold the easy dataitems
        # if self.threshold_byloss:
        #     th = self.threshold
        #     # loss_to_keep = loss[loss > th]
        #     loss_to_keep = loss[int(loss > th)]
        #     # print("loss_to_keep")
        #     # print(type(loss_to_keep))
        #     # print(loss_to_keep)
        #     # print(self.size(loss_to_keep))
        #     if self.size(loss_to_keep) > 0:
        #         loss = loss_to_keep.mean()
        # else:
        #     loss = loss.mean()

        loss = loss.mean()
        print("loss end = ", loss)
        return loss

if __name__ == '__main__':
    target = "Ascend"
    device_id = 5

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

    # from mindspore import Profiler
    #
    # # Init Profiler
    # # Note that the Profiler should be initialized before model.train
    # profiler = Profiler(output_path="./dataUnit21")

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

    # test add_speech_distortions
    network = SepFormer(hparams)
    net_with_loss = NetWithLoss(network, hparams)
    network.set_train(True)

    shape = (1, 134920)
    print("input shape = ", shape)
    id = Tensor(1, dtype=mstype.int32)
    uniformreal = ops.UniformReal(seed=2)
    mix_sig = uniformreal(shape)
    s1_sig = uniformreal(shape)
    s2_sig = uniformreal(shape)
    loss = net_with_loss(id, mix_sig, s1_sig, s2_sig)
    print("loss = ", loss)
    # print(predictions.shape)
    # print(targets.shape)
    # done PYNATIVE_MODE OK
    # (1, 45840)
    # (1, 45840, 2)
    # end test add_speech_distortions

    # profiler.analyse()
    print("============== Starting Training ==============")


"""

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

    # this is for test train_dataset:
    # label = ["id", "mix_sig", "s1_sig", "s2_sig"]
    # for _, data in enumerate(train_dataset.create_dict_iterator()):
    #     print(_)
    #     id = data["id"]
    #     mix_sig = data["mix_sig"]
    #     s1_sig = data["s1_sig"]
    #     s2_sig = data["s2_sig"]
    #     print("mix_sig.shape = ", mix_sig.shape)
    #     print("s1_sig.shape = ", s1_sig.shape)
    #     print("s2_sig.shape = ", s2_sig.shape)
    #     if _ > 10:
    #         break


    train_data_size = train_dataset.get_dataset_size()
    print("train_data_size = ", train_data_size)
    # train_data_size = 1

    # create network
    network = SepFormer(hparams)
    net_with_loss = NetWithLoss(network, hparams)
    network.set_train(True)

    # optimizer
    scale_manager = None
    loss_scale = 1.0
    lr = 0.00015
    # print("network.trainable_params()")
    # print(network.trainable_params())
    optimizer = nn.Adam(params=network.trainable_params(),
                        learning_rate=lr,
                        loss_scale=loss_scale)

    model = Model(net_with_loss, optimizer=optimizer, loss_scale_manager=scale_manager)

    # Monitor
    time_cb = TimeMonitor(data_size=train_data_size)
    loss_cb = LossMonitor()

    # save checkpoint
    ckpt_config = CheckpointConfig(save_checkpoint_steps=train_data_size,
                                   keep_checkpoint_max=5)
    ckpoint_cb = ModelCheckpoint(prefix='{}'.format(SepFormer),
                                 directory="./result/ckpt_{}/",
                                 config=ckpt_config)
    callbacks_list = [loss_cb, time_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    print(type(train_dataset))
    # model.train(2, train_dataset, callbacks=callbacks_list, dataset_sink_mode=True)
    model.train(2, train_dataset, callbacks=callbacks_list, dataset_sink_mode=False)
    print("============== Starting Training ==============")
"""