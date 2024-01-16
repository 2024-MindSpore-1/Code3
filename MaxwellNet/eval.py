# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com
import mindspore as ms
from mindspore import ops
from src.MaxwellNet import MaxwellNet

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import os
import json
import argparse


def main(args):
    # use GPU
    # ms.set_context(device_target="GPU")
    # use Ascend
    ms.set_context(device_target="Ascend")
    specs_filename = os.path.join(
        os.getcwd(), args.directory, 'specs_maxwell.json')
    specs = json.load(open(specs_filename))

    physical_specs = specs['PhysicalSpecs']
    Nx = physical_specs['Nx']
    Nz = physical_specs['Nz']
    dpl = physical_specs['dpl']
    wavelength = physical_specs['wavelength']
    symmetry_x = physical_specs['symmetry_x']
    mode = physical_specs['mode']

    delta = wavelength / dpl
    Nx = Nx * (symmetry_x + 1)

    scat_pot_np = np.load(os.path.join(
        args.directory, args.sample_filename))['sample']
    ri = np.load(os.path.join(args.directory,
                 args.sample_filename))['n']

    model_directory = os.path.join(args.directory, 'model', args.model_filename)

    model = MaxwellNet(**specs["NetworkSpecs"], **
                       specs["PhysicalSpecs"])
    ms.load_checkpoint(model_directory, model)
    model.set_train(False)

    scat_pot_ms = ms.Tensor(np.float32(scat_pot_np))
    ri_ms = ms.Tensor([np.float32(ri)])

    (diff, total) = model(scat_pot_ms, ri_ms)
    
    total_np = total.asnumpy()
    diff_np = diff.asnumpy()
    total_np = total_np[0, 0::2, :, :] + 1j*total_np[0, 1::2, :, :]
    diff_np = diff_np[0, 0::2, :, :] + 1j*diff_np[0, 1::2, :, :]
    scat_pot_np = scat_pot_np[0, :, :, :] * (ri-1) + 1

    # Here, I use the following min max values to present the data just for visualization. This is approximately correct.
    # Please note that the output values from MaxwellNet are defined on Yee grid.
    # So, if you want to quantitatively compare the MaxwellNet outputs with solutions from another solver, you should compare two solutions at the same Yee grid points.
    x_min = -(Nx//2) * delta
    x_max = (Nx//2-1) * delta
    z_min = -(Nz//2) * delta
    z_max = (Nz//2-1) * delta
    fontsize = 20

    if mode == 'te':
        if symmetry_x == True:
            scat_pot_np = np.pad(np.concatenate(
                (np.flip(scat_pot_np[0, 1::, :], 0), scat_pot_np[0, :, :]), 0), ((1, 0), (0, 0)))
            total_np = np.pad(np.concatenate(
                (np.flip(total_np[0, 1::, :], 0), total_np[0, :, :]), 0), ((1, 0), (0, 0)))
            diff_np = np.pad(np.concatenate(
                (np.flip(diff_np[0, 1::, :], 0), diff_np[0, :, :]), 0), ((1, 0), (0, 0)))

        fig, axs = plt.subplots(1, 2, figsize=(8, 5))
        fig.suptitle('TE mode - Sherical Lens', fontsize=fontsize)

        img0 = axs[0].imshow(scat_pot_np, extent=[
                             z_min, z_max, x_min, x_max], vmin=1, vmax=ri)
        axs[0].set_title('RI distribution', fontsize=fontsize)
        divider0 = make_axes_locatable(axs[0])
        cax0 = divider0.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img0, cax=cax0)

        img1 = axs[1].imshow(np.abs(total_np), extent=[
                             z_min, z_max, x_min, x_max])
        axs[1].set_title('Ey (envelop)', fontsize=fontsize)
        divider1 = make_axes_locatable(axs[1])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img1, cax=cax1)

        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), args.directory, 'te_result.png'))

    elif mode == 'tm':
        if symmetry_x == True:
            scat_pot_x_np = np.concatenate(
                (np.flip(scat_pot_np[0, :, :], 0), scat_pot_np[0, :, :]), 0)
            scat_pot_z_np = np.pad(np.concatenate(
                (np.flip(scat_pot_np[1, 1::, :], 0), scat_pot_np[1, :, :]), 0), ((1, 0), (0, 0)))
            total_x_np = np.concatenate(
                (np.flip(total_np[0, :, :], 0), total_np[0, :, :]), 0)
            total_z_np = np.pad(np.concatenate(
                (-np.flip(total_np[1, 1::, :], 0), total_np[1, :, :]), 0), ((1, 0), (0, 0)))

        fig, axs = plt.subplots(2, 2, figsize=(8, 10))
        fig.suptitle('TM mode - Sherical Lens', fontsize=fontsize)
        img00 = axs[0, 0].imshow(scat_pot_x_np, extent=[
                                 z_min, z_max, x_min, x_max], vmin=1, vmax=ri)
        axs[0, 0].set_title('RI distribution', fontsize=fontsize)
        divider00 = make_axes_locatable(axs[0, 0])
        cax00 = divider00.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img00, cax=cax00)

        img01 = axs[0, 1].imshow(np.abs(total_x_np), extent=[
                                 z_min, z_max, x_min, x_max])
        axs[0, 1].set_title('Ex (envelop)', fontsize=fontsize)
        divider01 = make_axes_locatable(axs[0, 1])
        cax01 = divider01.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img01, cax=cax01)

        img10 = axs[1, 0].imshow(scat_pot_z_np, extent=[
                                 z_min, z_max, x_min, x_max], vmin=1, vmax=ri)
        axs[1, 0].set_title('RI distribution', fontsize=fontsize)
        divider10 = make_axes_locatable(axs[1, 0])
        cax10 = divider10.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img10, cax=cax10)

        img11 = axs[1, 1].imshow(np.abs(total_z_np), extent=[
                                 z_min, z_max, x_min, x_max])
        axs[1, 1].set_title('Ez (envelop)', fontsize=fontsize)
        divider11 = make_axes_locatable(axs[1, 1])
        cax11 = divider11.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img11, cax=cax11)

        plt.tight_layout()
        plt.savefig(os.path.join(os.getcwd(), args.directory, 'tm_result.png'))

    else:
        raise KeyError("'mode' should me either 'te' or 'tm'.")


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Train a MaxwellNet")
    arg_parser.add_argument(
        "--directory",
        "-d",
        required=True,
        default='examples\spheric_te',
        help="This directory should include "
             + "'sample.npz' (input to MaxwellNet) and "
             + "'model' folder where trained 'MaxwellNet' parameters are saved and "
             + "'specs_maxwell.json' used during training."
    )
    arg_parser.add_argument(
        "--model_filename",
        required=True,
        help="This filename indicates the saved model file name within 'directory\model\'."
    )
    arg_parser.add_argument(
        "--sample_filename",
        required=True,
        help="This filename indicates a .npz file to be provied to MaxwellNet to calculate the solution, and it should be located in 'directory\'."
    )

    args = arg_parser.parse_args()
    main(args)
