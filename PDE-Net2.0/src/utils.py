import mindspore as ms
import yaml
import numpy as np
import os
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.dataset import Dataset, DataGenerator
from src.pdenet import PolyPDENet2D, relative_l2_error
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox


# def init_env(configs):
#     if configs['device_target'] not in ['CPU', 'GPU', 'Ascend']:
#         raise ValueError('invalid device target.')
#     ms.set_context(device_target=configs['device_target'])
#     print('----------------------------- Env settings -----------------------------')
#     print('running on {}'.format(configs['device_target']))
#
#     if configs['context_mode'] not in ['graph', 'pynative']:
#         raise ValueError('invalid context mode.')
#
#     if configs['context_mode'] == 'graph':
#         ms.set_context(mode=ms.GRAPH_MODE)
#     else:
#         ms.set_context(mode=ms.PYNATIVE_MODE)
#     print('context mode: {}'.format(configs['context_mode']))
#
#     if isinstance(configs['device_id'], int):
#         ms.set_context(device_id=configs['device_id'])
#     print('device id: {}\n'.format(configs['device_id']))
#     return


def init_env(env_args):
    ms.set_context(mode=ms.GRAPH_MODE if env_args.mode.upper().startswith("GRAPH") else ms.PYNATIVE_MODE,
                   save_graphs=env_args.save_graphs,
                   save_graphs_path=env_args.save_graphs_path,
                   device_target=env_args.device_target,
                   device_id=env_args.device_id)
    print('----------------------------- Env settings -----------------------------')
    print('running on {}, device id: {}, context mode: {}'.format(env_args.device_target, env_args.device_id, env_args.mode))
    return


def get_config(config_path):
    with open(config_path, 'r') as c_f:
        config = yaml.safe_load(c_f)
    config['mesh_size'] = (config['mesh_size'], config['mesh_size'])
    config['mesh_range'] = ((0, config['mesh_range'] * np.pi), (0, config['mesh_range'] * np.pi))
    config['sample_mesh_size'] = (config['sample_mesh_size'], config['sample_mesh_size'])
    config['dx'] = tuple((config['mesh_range'][i][1] - config['mesh_range'][i][0]) / config['sample_mesh_size'][i]
                         for i in range(len(config['sample_mesh_size'])))
    config['kernel_size'] = (config['kernel_size'], config['kernel_size'])
    config['lr_reduce_interval'] = config['blocks_num'] // config['lr_reduce_times']
    return config


def mkdir(config):
    if not os.path.exists(config['save_directory']):
        os.mkdir(config['save_directory'])
    if not os.path.exists(config['data_directory']):
        os.mkdir(config['data_directory'])
    if not os.path.exists(config['images_directory']):
        os.mkdir(config['images_directory'])
    test_predicts_f_name = 'test_predicts_num{}_step{}.csv'.format(config['test_data_num'],
                                                                   config['test_data_step_num'])
    test_predicts_path = os.path.join(config['data_directory'], test_predicts_f_name)
    if os.path.exists(test_predicts_path):
        os.remove(test_predicts_path)
    return


def init_model(config):
    pde_net = PolyPDENet2D(dt=config['dt'], dx=config['dx'], kernel_size=config['kernel_size'],
                           symnet_hidden_num=config['symnet_hidden_num'], symnet_init_range=config['symnet_init_range'],
                           max_order=config['max_order'], if_upwind=config['if_upwind'], dtype=ms.float32)
    return pde_net


def load_param_dict(save_directory: str, step: int, epoch: int):
    ckpt_path = os.path.join(save_directory, 'pde_net_step{}_epoch{}.ckpt'.format(step, epoch))
    param_dict = load_checkpoint(ckpt_file_name=ckpt_path)
    return param_dict


def generate_train_data(config, data_generator):
    trajectories = data_generator.generate_data(data_num=config['generate_data_num'], step_num=config['blocks_num'])
    train_dataset = Dataset(trajectories=trajectories, batch_size=config['batch_size'],
                            dtype=ms.float32, shuffle=True)
    return train_dataset


def evaluate(model: PolyPDENet2D, data_generator: DataGenerator, config: dict, step_num: int):
    model.set_train(mode=False)
    trajectories = data_generator.generate_data(data_num=config['evaluate_data_num'], step_num=step_num)
    # TNCHW
    trajectories = np.swapaxes(trajectories, 0, 1)
    trajectories = ms.Tensor(trajectories, dtype=ms.float32)
    former = trajectories[0]
    max_error = 0
    for step in range(1, step_num + 1):
        middle = model.construct(former, T=config['dt'])
        batch_error = relative_l2_error(predict=middle, label=trajectories[step])
        error = ms.ops.mean(batch_error)
        if error > max_error:
            max_error = error
        former = middle
    model.set_train(mode=True)
    return max_error


def _save_test_trajectories(config, test_trajectories: np.ndarray, test_data_path: str):
    r""" (data_num, step_num + 1, 2, sample_mesh_size_y, sample_mesh_size_x) >> NTCHW. """
    assert test_trajectories.shape[1:] == (config['test_data_step_num'] + 1, 2, config['sample_mesh_size'][0],
                                           config['sample_mesh_size'][1])
    trajectories = test_trajectories.reshape((-1, config['sample_mesh_size'][0], config['sample_mesh_size'][1]))
    image_list = [trajectories[i] for i in range(trajectories.shape[0])]
    test_images = np.concatenate(image_list, axis=0)
    with open(test_data_path, 'ab') as f:
        np.savetxt(f, test_images, delimiter='\t')
    return


def save_test_labels(config, test_labels: np.ndarray):
    test_labels_f_name = 'test_data_num{}_step{}.csv'.format(config['test_data_num'], config['test_data_step_num'])
    test_labels_path = os.path.join(config['data_directory'], test_labels_f_name)
    _save_test_trajectories(config=config, test_trajectories=test_labels, test_data_path=test_labels_path)
    return


def save_test_predicts(config, test_predicts: np.ndarray):
    test_predicts_f_name = 'test_predicts_num{}_step{}.csv'.format(config['test_data_num'], config['test_data_step_num'])
    test_predicts_path = os.path.join(config['data_directory'], test_predicts_f_name)
    _save_test_trajectories(config=config, test_trajectories=test_predicts, test_data_path=test_predicts_path)
    return


def generate_test_data(config, data_generator):
    print('generating data ...')
    generate_num = 10
    if generate_num > config['test_data_num']:
        generate_num = config['test_data_num']

    generate_sum = 0
    while generate_sum < config['test_data_num']:
        if generate_sum + generate_num > config['test_data_num']:
            generate_num = config['test_data_num'] - generate_sum
        trajectories = data_generator.generate_data(data_num=generate_num, step_num=config['test_data_step_num'])
        save_test_labels(config=config, test_labels=trajectories)
        generate_sum += generate_num
    print('finished.')
    return


def _load_results(result_path):
    errors = np.loadtxt(result_path, delimiter='\t')
    nan = np.isnan(errors).any(axis=1)
    errors = errors[~nan]
    return errors


def _load_test_trajectories(config, path, indices=None):
    test_images = np.loadtxt(fname=path, delimiter='\t')
    test_trajectories = np.split(test_images, indices_or_sections=config['sample_mesh_size'][0], axis=0)
    test_trajectories = np.stack(test_trajectories, axis=0)
    test_trajectories = test_trajectories.reshape((config['test_data_num'], config['test_data_step_num'] + 1, 2,
                                                   config['sample_mesh_size'][0], config['sample_mesh_size'][1]))
    if indices is None:
        indices = np.isnan(test_trajectories).any(axis=tuple(range(1, 5)))
    test_trajectories = test_trajectories[~indices]
    return test_trajectories, indices


def load_test_data(config):
    test_data_f_name = 'test_data_num{}_step{}.csv'.format(config['test_data_num'], config['test_data_step_num'])
    test_data_path = os.path.join(config['data_directory'], test_data_f_name)
    test_data, _ = _load_test_trajectories(config=config, path=test_data_path)
    test_dataset = Dataset(trajectories=test_data, batch_size=config['test_batch_size'],
                           dtype=ms.float32, shuffle=False)
    return test_dataset


def show_test_errors(config, show_results: bool):
    errors = _load_results(config['test_error_path'])
    p25 = np.percentile(errors, 25, axis=0)
    p75 = np.percentile(errors, 75, axis=0)
    image_fig = os.path.join(config['images_directory'], 'relative_error.png')
    steps = list(range(1, p25.shape[0] + 1))
    plt.figure()
    plt.plot(p25, color='tomato')
    plt.plot(p75, color='tomato')
    plt.fill_between(steps, p25, p75, alpha=0.5, color='tomato')
    plt.ylim(0.001, 10)
    plt.yscale('log')
    plt.xlabel('step')
    plt.ylabel('relative l2 error')
    plt.savefig(image_fig)
    print('================ Test Error Image saved as {} ================'.format(image_fig))
    if show_results:
        plt.show()
    return


def show_test_comparison(config, show_results: bool):
    total_time = config['dt'] * config['test_data_step_num']
    show_num = 5
    test_predicts_f_name = 'test_predicts_num{}_step{}.csv'.format(config['test_data_num'],
                                                                   config['test_data_step_num'])
    test_predicts_path = os.path.join(config['data_directory'], test_predicts_f_name)
    test_predicts, indices = _load_test_trajectories(config=config, path=test_predicts_path)
    test_data_f_name = 'test_data_num{}_step{}.csv'.format(config['test_data_num'], config['test_data_step_num'])
    test_data_path = os.path.join(config['data_directory'], test_data_f_name)
    test_labels, _ = _load_test_trajectories(config=config, path=test_data_path, indices=indices)
    show_interval = config['test_data_step_num'] // (show_num - 1)
    x_ticks = np.arange(0, config['sample_mesh_size'][0], 5)
    y_ticks = np.arange(0, config['sample_mesh_size'][1], 5)

    chs = ['u', 'v']

    def sub_imshow(fig, ax, name, ax_data):
        im = ax.imshow(ax_data, cmap=config['color_map'])
        ax.set_title('{} T={:.1f}'.format(name, show_idx * show_interval * config['dt']), fontsize=5, pad=2)
        ax.set_xticks(ticks=x_ticks)
        ax.tick_params(length=3, direction='in')
        ax.set_xticklabels([])
        ax.set_yticks(ticks=y_ticks)
        ax.set_yticklabels([])

        ax_pos = ax.get_position()
        pad = 0.005
        width = 0.005
        cax_pos = Bbox.from_extents(ax_pos.x1 + pad, ax_pos.y0, ax_pos.x1 + pad + width, ax_pos.y1)
        cax = ax.figure.add_axes(cax_pos)
        c_bar = fig.colorbar(im, cax=cax)
        c_bar.ax.tick_params(labelsize=3, length=1, pad=1)
        return

    plt.figure(figsize=(6, show_num))
    figs, axes = plt.subplots(6, show_num)
    plt.subplots_adjust(wspace=0.2, hspace=0.3)
    for ch_idx in range(len(chs)):
        ch_name = chs[ch_idx]
        for show_idx in range(show_num):
            label = test_labels[0, show_idx * show_interval, ch_idx]
            sub_imshow(fig=figs, ax=axes[0 + ch_idx * 3][show_idx], name=ch_name + ' label', ax_data=label)

            predict = test_predicts[0, show_idx * show_interval, ch_idx]
            sub_imshow(fig=figs, ax=axes[1 + ch_idx * 3][show_idx], name=ch_name + ' predict', ax_data=predict)

            error = predict - label
            sub_imshow(fig=figs, ax=axes[2 + ch_idx * 3][show_idx], name=ch_name + ' error', ax_data=error)

    image_fig = os.path.join(config['images_directory'], 'comparison.png')
    plt.savefig(image_fig, dpi=500)
    print('================ Test Comparison Image saved as {} ================'.format(image_fig))
    if show_results:
        plt.show()
    return


def test(config, show_results: bool = False):
    data_generator = DataGenerator(config=config)
    test_data_f_name = 'test_data_num{}_step{}.csv'.format(config['test_data_num'], config['test_data_step_num'])
    if not os.path.exists(os.path.join(config['data_directory'], test_data_f_name)):
        generate_test_data(config=config, data_generator=data_generator)

    test_dataset = load_test_data(config=config)
    pde_net = init_model(config=config)
    param_dict = load_param_dict(save_directory=config['save_directory'], step=config['blocks_num'],
                                 epoch=config['epochs'])
    param_not_load = load_param_into_net(net=pde_net, parameter_dict=param_dict)

    if len(param_not_load) == 0:
        print('=============== Net saved at last step is loaded. ===============')
    else:
        print('!!!!!!!!! param not loaded: ', param_not_load)

    error_list = []
    pde_net.set_train(mode=False)
    data_num = 0
    for batch_trajectory in test_dataset.fetch():
        data_num += batch_trajectory.shape[0]
        batch_trajectory = ms.numpy.swapaxes(batch_trajectory, 0, 1)
        batch_former = batch_trajectory[0]
        step_error_list = []
        predict_list = [batch_former.asnumpy()]
        for step in range(1, config['test_data_step_num'] + 1):
            print('predicting data_num {} step {} ...'.format(data_num, step))
            batch_mid = pde_net.construct(batch_former, T=config['dt'])
            batch_label = batch_trajectory[step]
            batch_error = relative_l2_error(predict=batch_mid, label=batch_label)
            step_error_list.append(batch_error.asnumpy())
            predict_list.append(batch_mid.asnumpy())
            batch_former = batch_mid
        step_error_list = np.stack(step_error_list, axis=1)
        predict_list = np.stack(predict_list, axis=1)
        save_test_predicts(config=config, test_predicts=predict_list)
        error_list.append(step_error_list)
    # test data num * test step num
    error_list = np.concatenate(error_list, axis=0)
    np.savetxt(fname=config['test_error_path'], X=error_list, delimiter='\t')
    print('========================= expression of trained model =========================')
    pde_net.show_expression()
    show_test_errors(config=config, show_results=show_results)
    show_test_comparison(config=config, show_results=show_results)
    return
