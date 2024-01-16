import os
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pylab import minorticks_on




def set_font(small_size=25, medium_size=28, bigger_size=45):
    """set plot font"""
    font_legend = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 18,
        'style': 'italic'
    }
    font_title = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 22
    }
    matplotlib.rcParams['mathtext.default'] = 'regular'
    plt.rc('font', size=small_size)  # controls default text sizes
    plt.rc('axes', titlesize=small_size)  # fontsize of the axes title
    plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=small_size)  # fontsize of the tick labels
    plt.rc('legend', fontsize=bigger_size)  # legend fontsize
    plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
    plt.rc('figure', titlesize=medium_size)  # fontsize of the figure title
    return font_title, font_legend

def plot_u_v_p(eval_dataset, model, post_dir,batch_size,grid_path):
    """plot u v p image"""
    set_font(small_size=15)
    for data in eval_dataset.create_dict_iterator(output_numpy=False):
        inputs = data["inputs"]
        labels = data["labels"]
        pred = model(inputs)
        inputs = inputs.asnumpy()
        print("shape of inputs {} type {} max {}".format(inputs.shape, type(inputs), inputs.max()))
        print("shape of labels {} type {} max {}".format(labels.shape, type(labels), labels.max()))
        print("shape of pred {} type {} max {}".format(pred.shape, type(pred), pred.max()))
        break
    save_img_dir = os.path.join(post_dir, 'uvp_ViT')
    print(f'save img dir: {save_img_dir}')
    model_name = "ViT_"
    for i in range(batch_size):
        print("plot {} / {} done".format(i + 1, batch_size))
        plot_contourf(labels, pred, i, save_img_dir, grid_path, model_name)

def plot_config(xgrid, ygrid, data, index, title_name=None):
    """plot_config"""
    set_font()
    fig_note = f"({chr(ord('a')+index-1)})"
    plt.title(title_name, y=0.8, fontsize=20, fontweight=500)
    if index in (1, 4, 7):
        plt.ylabel('y/d', fontsize=18, style='italic')
    if index >= 7:
        plt.xlabel('x/d', fontsize=18, style='italic')
        min_value, max_value = np.min(data), 0.6 * np.max(data)
    else:
        min_value, max_value = 0.9 * np.min(data), 0.9 * np.max(data)
    plt.yticks(fontsize=15)
    plt.xticks(fontsize=15)
    plt.xlim((-0.5, 2))
    plt.ylim((-0.5, 2))
    box = {'facecolor': 'w', 'edgecolor': 'w'}
    plt.text(-0.43, 1.65, fig_note, bbox=box, fontsize=18)
    plt.contour(xgrid, ygrid, data, 21, vmin=min_value, vmax=max_value,
                linestyles="dashed", alpha=0.2)
    h = plt.contourf(xgrid, ygrid, data, 21,
                     vmin=min_value, vmax=max_value, cmap='jet')

    cb1 = plt.colorbar(h, fraction=0.03, pad=0.05)
    cb1.ax.tick_params(labelsize=13)
    tick_locator = ticker.MaxNLocator(nbins=4)
    cb1.locator = tick_locator
    cb1.update_ticks()
    minorticks_on()
    return h

def plot_contourf(label, pred, num, save_img_dir, grid_path, model_name="ViT"):
    """plot_contourf"""
    label = label.asnumpy()
    pred = pred.asnumpy()
    save_uvp_name = f"{save_img_dir}/{str(num).rjust(3, '0')}_UVP.png"
    save_cp_name = f"{save_img_dir}/{str(num).rjust(3, '0')}_cp.png"
    grid = np.load(grid_path)[num, ...]
    xgrid, ygrid = grid[..., 0], grid[..., 1]
    fig = plt.figure(1, figsize=(20, 14))

    for i, map_name in enumerate(['U', 'V', 'P']):
        gt = label[num][:, :, i]
        pd = pred[num][:, :, i]
        l1_error = np.abs(gt - pd)
        k1, k2, k3 = i + 1, i + 4, i + 7
        plt.subplot(3, 3, k1)
        plot_config(xgrid, ygrid, gt, k1, title_name=f'CFD-{map_name}')
        plt.subplot(3, 3, k2)
        plot_config(xgrid, ygrid, pd, k2, title_name=f'{model_name}-{map_name}')
        plt.subplot(3, 3, k3)
        plot_config(xgrid, ygrid, l1_error, k3, title_name=f'l1-{map_name}')
    plt.subplots_adjust(wspace=0.25, hspace=0.2)
    plt.show()
    fig.savefig(save_uvp_name, bbox_inches='tight', pad_inches=0.15)
    plt.close()

    # cp plot
    label_single = label[num][:, :, -1]
    pred_single = pred[num][:, :, -1]
    fig = plt.figure(2, figsize=(5, 5))
    plt.scatter(xgrid[0, :], -label_single[0, :], s=30, facecolors='none', edgecolors='k', label='CFD')
    plt.plot(xgrid[0, :], -pred_single[0, :], 'b', linewidth=2, label=model_name)
    plt.ylabel('cp', style='italic', fontsize=18)
    plt.xlabel('x/d', style='italic', fontsize=18)
    plt.legend()
    plt.show()
    fig.savefig(save_cp_name, bbox_inches='tight', pad_inches=0.05)
    plt.close()