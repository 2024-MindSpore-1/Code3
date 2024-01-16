import argparse
import os
import mindspore as ms
import numpy as np
from mindspore import nn
from mindspore.amp import all_finite
from src.dataset import DataGenerator
from src.pdenet import PDENetWithLoss
from src.utils import init_env, get_config, init_model, load_param_dict, mkdir, generate_train_data
from src.utils import test, evaluate
from mindspore.train.serialization import load_param_into_net


ms.set_seed(1999)
np.random.seed(1999)


def single_train(step_num: int, config: dict, data_generator: DataGenerator):
    # generate data: (data_num, step_num + 1, 2, sample_mesh_size_y, sample_mesh_size_x)
    train_dataset = generate_train_data(config=config, data_generator=data_generator)
    pde_net = init_model(config=config)
    if step_num == 1:
        # warm up
        regularization = False
        frozen = True
        pde_net.set_frozen(frozen=frozen)
        epochs = config['warmup_epochs']
        lr = config['warmup_lr']
    else:
        if step_num == 2:
            load_epoch = config['warmup_epochs']
        else:
            load_epoch = config['epochs']

        param_dict = load_param_dict(save_directory=config['save_directory'], step=step_num - 1, epoch=load_epoch)
        param_not_load = load_param_into_net(net=pde_net, parameter_dict=param_dict)
        print('=============== Net saved at last step is loaded. ===============')
        print('!!!!!!!!! param not loaded: ', param_not_load)

        regularization = True
        frozen = False
        pde_net.set_frozen(frozen=frozen)
        epochs = config['epochs']
        lr = config['lr'] * np.power(config['lr_reduce_gamma'], (step_num - 1) // config['lr_reduce_interval'])

    # lr scheduler
    my_optimizer = nn.Adam(params=pde_net.trainable_params(), learning_rate=lr)
    net_with_loss = PDENetWithLoss(pde_net=pde_net,
                                   moment_loss_threshold=config['moment_loss_threshold'],
                                   symnet_loss_threshold=config['symnet_loss_threshold'],
                                   moment_loss_scale=config['moment_loss_scale'],
                                   symnet_loss_scale=config['symnet_loss_scale'],
                                   step_num=step_num, regularization=regularization)

    def forward_fn(trajectory):
        loss = net_with_loss.get_loss(batch_trajectory=trajectory)
        return loss
    value_and_grad = ms.ops.value_and_grad(forward_fn, None, weights=my_optimizer.parameters)

    def train_process(trajectory):
        # TNCHW
        trajectory = ms.numpy.swapaxes(trajectory, 0, 1)
        loss, grads = value_and_grad(trajectory)
        if config['device_target'].upper() == 'ASCEND':
            status = ms.numpy.zeros((8, ))
        else:
            status = None
        if all_finite(grads, status=status):
            my_optimizer(grads)
        return loss

    for epoch_idx in range(1, epochs + 1):
        pde_net.set_train(mode=True)
        avg_loss = 0
        for batch_trajectory in train_dataset.fetch():
            train_loss = train_process(batch_trajectory)
            avg_loss += train_loss.asnumpy()
        print('step_num: {} -- epoch: {} -- lr: {} -- loss: {}'.format(step_num, epoch_idx,
                                                                       my_optimizer.learning_rate.value(),
                                                                       avg_loss))
        # generate new data
        if epoch_idx % config['generate_data_interval'] == 0:
            train_dataset = generate_train_data(config=config, data_generator=data_generator)
        # evaluate
        if epoch_idx % config['evaluate_interval'] == 0:
            evaluate_error = evaluate(model=pde_net, data_generator=data_generator, config=config, step_num=step_num)
            print('=============== Max evaluate error: {} ==============='.format(evaluate_error))
            print('=============== Current Expression ===============')
            pde_net.show_expression(coe_threshold=config['coe_threshold'])

        if epoch_idx == epochs:
            print('=============== Current Expression ===============')
            pde_net.show_expression(coe_threshold=config['coe_threshold'])
            pde_net.show_kernels()
            save_path = os.path.join(config['save_directory'],
                                     'pde_net_step{}_epoch{}.ckpt'.format(step_num, epoch_idx))
            ms.save_checkpoint(pde_net, save_path)
    return


def train(config):
    data_generator = DataGenerator(config=config)
    for step_num in range(2, config['blocks_num'] + 1):
        single_train(config=config, step_num=step_num, data_generator=data_generator)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="burgers PDENet2.0 train")
    parser.add_argument("--mode", type=str, default="PYNATIVE", choices=["PYNATIVE"], help="Running in PYNATIVE_MODE")
    parser.add_argument("--save_graphs", type=bool, default=False, choices=[True, False],
                        help="Whether to save intermediate compilation graphs")
    parser.add_argument("--save_graphs_path", type=str, default="./graphs")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["CPU", "GPU", "Ascend"],
                        help="The target device to run, support 'CPU', 'Ascend', 'GPU'")
    parser.add_argument("--device_id", type=int, default=0, help="ID of the target device")
    parser.add_argument("--config_file_path", type=str, default="./config.yaml")
    args = parser.parse_args()
    init_env(env_args=args)
    my_config = get_config(args.config_file_path)
    my_config['device_target'] = args.device_target
    my_config['context_mode'] = args.mode
    my_config['device_id'] = args.device_id

    mkdir(config=my_config)
    train(config=my_config)
    test(config=my_config)
