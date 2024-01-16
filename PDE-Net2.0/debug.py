import argparse
import os.path
from src.dataset import DataGenerator
from src.pdenet import PolyPDENet2D, relative_l2_error
import numpy as np
import matplotlib.pyplot as plt
import mindspore as ms
from src.pdenet.utils import _M2K
from src.utils import init_env, get_config


def debug_generator():
    config = get_config('config.yaml')
    config['mesh_size'] = (128, 128)
    config['mesh_range'] = ((0, 2 * np.pi), (0, 2 * np.pi))
    config['max_dt'] = 1e-2 / 16
    config['max_freq'] = 4
    config['viscosity'] = 0.05
    config['init_shift'] = 2.
    config['init_range'] = 2.
    config['sample_mesh_size'] = (32, 32)
    config['start_noise_level'] = 0.001
    config['end_noise_level'] = 0.001
    config['dt'] = 1e-2
    show_interval = 100
    show_num = 4
    step_num = show_interval * (show_num - 1) + 1

    if not os.path.exists('./images'):
        os.mkdir('./images')

    gen = DataGenerator(config)
    # NTCHW
    trajectory, raw_trajectory = gen.generate_data(data_num=1, step_num=step_num, require_raw_data=True,
                                                   dtype=np.float32)
    fig, axes = plt.subplots(nrows=4, ncols=show_num)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    xticks = np.arange(0, config['sample_mesh_size'][0], 5)
    yticks = np.arange(0, config['sample_mesh_size'][1], 5)
    raw_xticks = np.arange(0, config['mesh_size'][0], 25)
    raw_yticks = np.arange(0, config['mesh_size'][1], 25)
    for step in range(show_num):
        axes[0, step].imshow(trajectory[0, step * show_interval, 0], cmap=config['color_map'])
        axes[0, step].set_title('sampled u at step {}'.format(step * show_interval), fontsize=5)
        axes[0, step].set_xticks(xticks)
        axes[0, step].set_xticklabels(xticks, fontsize=5)
        axes[0, step].set_yticks(yticks)
        axes[0, step].set_yticklabels(yticks, fontsize=5)
        axes[1, step].imshow(raw_trajectory[0, step * show_interval, 0], cmap=config['color_map'])
        axes[1, step].set_title('raw u at step {}'.format(step * show_interval), fontsize=5)
        axes[1, step].set_xticks(raw_xticks)
        axes[1, step].set_xticklabels(raw_xticks, fontsize=5)
        axes[1, step].set_yticks(raw_yticks)
        axes[1, step].set_yticklabels(raw_yticks, fontsize=5)

        axes[2, step].imshow(trajectory[0, step * show_interval, 1], cmap=config['color_map'])
        axes[2, step].set_title('sampled v at step {}'.format(step * show_interval), fontsize=5)
        axes[2, step].set_xticks(xticks)
        axes[2, step].set_xticklabels(xticks, fontsize=5)
        axes[2, step].set_yticks(yticks)
        axes[2, step].set_yticklabels(yticks, fontsize=5)
        axes[3, step].imshow(raw_trajectory[0, step * show_interval, 1], cmap=config['color_map'])
        axes[3, step].set_title('raw v at step {}'.format(step * show_interval), fontsize=5)
        axes[3, step].set_xticks(raw_xticks)
        axes[3, step].set_xticklabels(raw_xticks, fontsize=5)
        axes[3, step].set_yticks(raw_yticks)
        axes[3, step].set_yticklabels(raw_yticks, fontsize=5)
    plt.savefig('./images/pde_solver_generate.png', dpi=300)


def manual_set_net(model: PolyPDENet2D):
    kernel_size = (5, 5)
    assert model.kernel_size == kernel_size
    mk_conversion = _M2K(shape=kernel_size)
    k01 = ms.numpy.zeros(kernel_size)
    k10 = ms.numpy.zeros(kernel_size)
    k02 = ms.numpy.zeros(kernel_size)
    k20 = ms.numpy.zeros(kernel_size)
    k01[2, 2] = -1.5
    k01[3, 2] = 2
    k01[4, 2] = -0.5

    k10[2, 2] = -1.5
    k10[2, 3] = 2
    k10[2, 4] = -0.5

    k02[1, 2] = 1
    k02[2, 2] = -2
    k02[3, 2] = 1

    k20[2, 1] = 1
    k20[2, 2] = -2
    k20[2, 3] = 1

    m01 = mk_conversion.k2m(k01)
    m10 = mk_conversion.k2m(k10)
    m02 = mk_conversion.k2m(k02)
    m20 = mk_conversion.k2m(k20)

    # set moment:
    model.name_cells()['fd_x0_y1'].free_moment.set_data(m01)
    model.name_cells()['fd_x1_y0'].free_moment.set_data(m10)
    model.name_cells()['fd_x0_y2'].free_moment.set_data(m02)
    model.name_cells()['fd_x2_y0'].free_moment.set_data(m20)

    u_weight_0 = ms.numpy.zeros((2, 12))
    u_weight_0[0][0] = 1
    u_weight_0[1][2] = 1
    u_weight_1 = ms.numpy.zeros((2, 13))
    u_weight_1[0][1] = 1
    u_weight_1[1][6] = 1
    u_weight_out = ms.numpy.zeros((1, 14))
    u_weight_out[0][3] = 0.05
    u_weight_out[0][5] = 0.05
    u_weight_out[0][12] = -1
    u_weight_out[0][13] = -1

    v_weight_0 = ms.numpy.zeros((2, 12))
    v_weight_0[0][6] = 1
    v_weight_0[1][7] = 1
    v_weight_1 = ms.numpy.zeros((2, 13))
    v_weight_1[0][0] = 1
    v_weight_1[1][8] = 1
    v_weight_out = ms.numpy.zeros((1, 14))
    v_weight_out[0][9] = 0.05
    v_weight_out[0][11] = 0.05
    v_weight_out[0][12] = -1
    v_weight_out[0][13] = -1

    model.name_cells()['symnet_dim0'].name_cells()['hidden_0'].weight.set_data(u_weight_0)
    model.name_cells()['symnet_dim0'].name_cells()['hidden_1'].weight.set_data(u_weight_1)
    model.name_cells()['symnet_dim0'].name_cells()['output'].weight.set_data(u_weight_out)

    model.name_cells()['symnet_dim1'].name_cells()['hidden_0'].weight.set_data(v_weight_0)
    model.name_cells()['symnet_dim1'].name_cells()['hidden_1'].weight.set_data(v_weight_1)
    model.name_cells()['symnet_dim1'].name_cells()['output'].weight.set_data(v_weight_out)
    print('Manually set pde-net expression: ')
    model.show_expression(coe_threshold=0.005)
    model.show_kernels()
    return model


def debug_pde_net():
    config = get_config('config.yaml')
    config['mesh_size'] = (128, 128)
    config['mesh_range'] = ((0, 2 * np.pi), (0, 2 * np.pi))
    config['max_order'] = 2
    config['max_dt'] = 1e-2 / 16
    config['max_freq'] = 4
    config['viscosity'] = 0.05
    config['init_shift'] = 2.
    config['init_range'] = 2.
    config['sample_mesh_size'] = (32, 32)
    config['start_noise_level'] = 0.001
    config['end_noise_level'] = 0.001
    config['dt'] = 1e-2
    config['kernel_size'] = 5
    config['symnet_hidden_num'] = 2

    show_interval = 1
    show_num = 4
    step_num = show_interval * (show_num -1) + 1

    if not os.path.exists('./images'):
        os.mkdir('./images')

    handcraft_net = PolyPDENet2D(dt=config['dt'], dx=config['dx'], max_order=config['max_order'],
                                 kernel_size=config['kernel_size'], symnet_hidden_num=config['symnet_hidden_num'],
                                 if_upwind=True, dtype=ms.float32)

    # manually set params:
    handcraft_net = manual_set_net(handcraft_net)

    # generate data:
    gen = DataGenerator(config)
    # NTCHW
    trajectory = gen.generate_data(data_num=1, step_num=step_num, dtype=np.float32)
    # TNCHW
    trajectory = np.swapaxes(trajectory, 0, 1)
    trajectory = ms.Tensor(trajectory, dtype=ms.float32)
    print('data generated.')

    # predict
    init = trajectory[0]
    net_trajectory = [init, ]
    former = init
    error_list = []
    for step in range(1, step_num + 1):
        print('predicting step {}'.format(step))
        latter = handcraft_net.construct(former, config['dt'])
        error = relative_l2_error(predict=latter, label=trajectory[step])
        error_list.append(error.asnumpy())
        net_trajectory.append(latter)
        former = latter

    net_trajectory = ms.ops.stack(net_trajectory, axis=0)
    # compare:
    plt.figure(figsize=(4, show_num))
    fig, axes = plt.subplots(nrows=4, ncols=show_num)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    xticks = np.arange(0, config['sample_mesh_size'][0], 5)
    yticks = np.arange(0, config['sample_mesh_size'][1], 5)
    for step in range(show_num):
        axes[0, step].imshow(trajectory[step * show_interval, 0, 0].asnumpy(), cmap=config['color_map'])
        axes[0, step].set_title('label u at step {}'.format(step * show_interval), fontsize=6)
        axes[0, step].set_xticks(xticks)
        axes[0, step].set_xticklabels(xticks, fontsize=5)
        axes[0, step].set_yticks(yticks)
        axes[0, step].set_yticklabels(yticks, fontsize=5)
        axes[1, step].imshow(net_trajectory[step * show_interval, 0, 0].asnumpy(), cmap=config['color_map'])
        axes[1, step].set_title('predict u at step {}'.format(step * show_interval), fontsize=6)
        axes[1, step].set_xticks(xticks)
        axes[1, step].set_xticklabels(xticks, fontsize=5)
        axes[1, step].set_yticks(yticks)
        axes[1, step].set_yticklabels(yticks, fontsize=5)

        axes[2, step].imshow(trajectory[step * show_interval, 0, 1].asnumpy(), cmap=config['color_map'])
        axes[2, step].set_title('label v at step {}'.format(step * show_interval), fontsize=6)
        axes[2, step].set_xticks(xticks)
        axes[2, step].set_xticklabels(xticks, fontsize=5)
        axes[2, step].set_yticks(yticks)
        axes[2, step].set_yticklabels(yticks, fontsize=5)
        axes[3, step].imshow(net_trajectory[step * show_interval, 0, 1].asnumpy(), cmap=config['color_map'])
        axes[3, step].set_title('predict v at step {}'.format(step * show_interval), fontsize=6)
        axes[3, step].set_xticks(xticks)
        axes[3, step].set_xticklabels(xticks, fontsize=5)
        axes[3, step].set_yticks(yticks)
        axes[3, step].set_yticklabels(yticks, fontsize=5)
    plt.savefig('./images/handcraft_pdenet_compare.png', dpi=300)
    plt.cla()
    plt.figure()
    plt.plot(error_list)
    plt.ylabel('relative error')
    plt.xlabel('step')
    plt.savefig('./images/handcraft_pdenet_relative_error.png')


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
    my_config = get_config(args.config_file_path)
    my_config['device_target'] = args.device_target
    my_config['context_mode'] = args.mode
    my_config['device_id'] = args.device_id
    init_env(env_args=args)
    debug_generator()
    debug_pde_net()
