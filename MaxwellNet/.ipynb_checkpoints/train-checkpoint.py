# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

# import torch
import mindspore as ms
import mindspore.ops as ops
from src.Dataset import LensDataset
from mindspore.nn import piecewise_constant_lr
from mindspore import set_seed
from src.MaxwellNet import MaxwellNet
from mindspore.ops import functional as F
import numpy as np
import random
import logging
import argparse
import os
import json
from datetime import datetime



transpose = ops.Transpose()

def main(directory, load_ckpt):
    ms.set_context( device_target="GPU")
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename=os.path.join(
                            os.getcwd(), directory, f"maxwellnet_{datetime.now():%Y-%m-%d %H-%M-%S}.log"),
                        filemode='w')

    logging.info("training " + directory)

    specs_filename = os.path.join(directory, 'specs_maxwell.json')

    if not os.path.isfile(specs_filename):
        raise Exception(
            'The experiment directory does not include specifications file "specs_maxwell.json"'
        )

    specs = json.load(open(specs_filename))

    seed_number = get_spec_with_default(specs, "Seed", None)

    set_seed(seed_number)
    rank = 0

    logging.info("Experiment description: \n" +
                 ' '.join([str(elem) for elem in specs["Description"]]))

    model = MaxwellNet(**specs["NetworkSpecs"], **specs["PhysicalSpecs"])

    logging.debug(specs["NetworkSpecs"])
    logging.debug(specs["PhysicalSpecs"])
    milestone = []
    learning_rates = [get_spec_with_default(specs, "LearningRate", 0.0001)]
    for i in range(1, int(get_spec_with_default(specs, "Epochs", 1)/get_spec_with_default(specs, "LearningRateDecayStep", 10000))+1):
        milestone.append(i*get_spec_with_default(specs, "LearningRateDecayStep", 10000))
        learning_rates.append(get_spec_with_default(specs, "LearningRate", 0.0001)*get_spec_with_default(specs, "LearningRateDecay", 1.0)**i)
    learning_rates.pop()

    lr = piecewise_constant_lr(milestone, learning_rates)

    optimizer = ms.nn.Adam(model.get_parameters(), learning_rate=lr, weight_decay=0)

    batch_size = get_spec_with_default(specs, "BatchSize", 1)
    epochs = get_spec_with_default(specs, "Epochs", 1)
    snapshot_freq = specs["SnapshotFrequency"]
    physical_specs = specs["PhysicalSpecs"]
    symmetry_x = physical_specs['symmetry_x']
    mode = physical_specs['mode']
    high_order = physical_specs['high_order']

    checkpoints = list(range(snapshot_freq, epochs + 1, snapshot_freq))

    filename = 'maxwellnet_' + mode + '_' + high_order

    train_dataset = LensDataset(directory, 'train')
    train_loader = ms.dataset.GeneratorDataset(train_dataset, ['sample', 'n', 'idx'], shuffle=True)
    
    logging.info("Train Dataset length: {}".format(len(train_dataset)))
    zeros = ops.Zeros()
    loss_train = zeros(
        (int(epochs),), ms.float32)

    if len(train_dataset) > 1:
        perform_valid = True
    else:
        perform_valid = False

    if perform_valid == True:
        valid_dataset = LensDataset(directory, 'valid')
        valid = valid_dataset.batch(batch_size=batch_size)
        valid_loader = valid.create_dict_iterator()
        logging.info("Valid Dataset length: {}".format(len(valid_dataset)))
        loss_valid = zeros(
            (int(epochs),), ms.float32)

    if load_ckpt is not None:
        optimizer.load_state_dict(ckpt_dict['optimizer'])
        scheduler.load_state_dict(ckpt_dict['scheduler'])
        loss_train[:ckpt_epoch:] = ckpt_dict['loss_train'][:ckpt_epoch:]
        logging.info("Check point loaded from {}-epoch".format(ckpt_epoch))

        start_epoch = ckpt_epoch
    else:
        start_epoch = 0
    loss = LossNet(model)
    train_net = MyTrainOneStepCell(loss, optimizer)
    logging.info("Training start")
    
    for epoch in range(start_epoch + 1, epochs + 1):
        train(train_loader, train_net, optimizer, epoch, loss_train,
              mode, symmetry_x)
        logging.info("[Train] {} epoch. Loss: {:.5f}".format(
            epoch, float(loss_train[epoch-1]))) if rank == 0 else None
        if perform_valid:
            valid(valid_loader, model, epoch, loss_valid,
                  mode, symmetry_x)
            logging.info("[Valid] {} epoch. Loss: {:.5f}".format(
                epoch, float(loss_valid[epoch-1]))) if rank == 0 else None

        if epoch in checkpoints:
            logging.info("Checkpoint saved at {} epoch.".format(
                epoch)) if rank == 0 else None
            if rank == 0:
                save_checkpoint(model
                   , directory, str(epoch) + '_' + mode + '_' + high_order)

        if epoch % 200 == 0:
            logging.info("'latest' checkpoint saved at {} epoch.".format(
                epoch)) if rank == 0 else None
            if rank == 0:
                save_checkpoint(model
                    , directory, 'latest')


_grad_scale = ops.MultitypeFuncGraph("grad_scale")

class MyTrainOneStepCell(ms.nn.TrainOneStepCell):
    def __init__(self, network, optimizer, grad_clip=True):
        super().__init__(network, optimizer)
        self.grad_clip = grad_clip

    def construct(self, *input): 
        loss = self.network(*input)
        sens = F.fill(loss.dtype, loss.shape, self.sens)

        grads =  self.grad(self.network, self.weights)(*input, sens)

        if self.grad_clip:
            grads = ops.clip_by_global_norm(grads, clip_norm=1e-3)
        grads = self.grad_reducer(grads)
        loss = F.depend(loss, self.optimizer(grads))
        return loss
    
class LossNet(ms.nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def construct(self, scat_pot_ms, ri_value_ms):

        (diff, total) = self.net(scat_pot_ms, ri_value_ms)
        l2 = diff.pow(2)
        loss = ms.ops.mean(l2)
        return loss
    
def train(train_loader, train_net, optimizer, epoch, loss_train, mode, symmetry):

    train_net.set_train()
    count = 0
    for data in train_loader:
        expand_dims = ops.ExpandDims()
        scat_pot_ms = expand_dims(data[0], 0)
        ri_value_ms = expand_dims(data[1], 0)
        
        loss_v = train_net(scat_pot_ms, ri_value_ms)
        loss_train[epoch-1] = loss_v




def valid(valid_loader, model, epoch, loss_valid, mode, symmetry):
    model.set_train(False)
    count = 0

    for data in valid_loader:
        
        expand_dims = ops.ExpandDims()
        scat_pot_ms = expand_dims(data[0], 0)
        ri_value_ms = expand_dims(data[1], 0)

        (diff, total) = model(scat_pot_ms,
                              ri_value_ms)  # [N, 1, H, W, D]

        l2 = diff.pow(2)
        loss = ms.mean(l2)

        loss_valid[epoch-1] = loss 



def save_checkpoint(state, directory, filename):
    model_directory = os.path.join(directory, 'model')
    if os.path.exists(model_directory) == False:
        os.makedirs(model_directory)
    ms.save_checkpoint(state, os.path.join(model_directory, filename))



def get_spec_with_default(specs, key, default):
    try:
        return specs[key]
    except KeyError:
        return default


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train a MaxwellNet")
    arg_parser.add_argument(
        "--directory",
        "-d",
        required=True,
        default='examples\spheric_te',
        help="This directory should include "
             + "all the training and network parameters in 'specs_maxwell.json', and logging will be "
             + "done in this directory as well.",
    )
    arg_parser.add_argument(
        "--load_ckpt",
        "-l",
        default=None,
        help="This should specify a filename of your checkpoint within 'directory'\model if you want to continue your training from the checkpoint.",
    )

    args = arg_parser.parse_args()
    main(args.directory, args.load_ckpt)
