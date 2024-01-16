# Copyright (c) 2022 Joowon Lim, limjoowon@gmail.com

import mindspore as ms
from mindspore import nn, Tensor
from .UNet import UNet
from mindspore import ops
import math
import numpy as np

transpose = ops.Transpose()
class MaxwellNet(nn.Cell):
    def __init__(self, depth=6, filter=16, norm='weight', up_mode='upconv',
                 wavelength=1, dpl=20, Nx=256, Nz=256, pml_thickness=16, symmetry_x=False, mode='te', high_order='fourth'):

        super(MaxwellNet, self).__init__()
        self.mode = mode
        if mode == 'te':
            in_channels = 1
            out_channels = 2
        elif mode == 'tm':
            in_channels = 2
            out_channels = 4

        self.high_order = high_order
        self.model = UNet(in_channels, out_channels,
                          depth, filter, norm, up_mode)

        # pixel size [um / pixel]
        delta = wavelength / dpl
        # wave-number [1 / um]
        k = 2 * math.pi / wavelength
        self.delta = ms.Tensor(
            delta, dtype=ms.float32, const_arg=True)
        self.k = ms.Tensor(
            k, dtype=ms.float32, const_arg=True)

        self.symmetry_x = symmetry_x

        if self.high_order == 'second':
            pad = 2
            self.pad = pad
        elif self.high_order == 'fourth':
            pad = 4
            self.pad = pad

        self.padding_ref = nn.SequentialCell([nn.ReflectionPad2d(
            (0, 0, pad, 0)), nn.ZeroPad2d((pad, pad, 0, pad))])
        self.padding_zero = nn.SequentialCell([nn.ZeroPad2d((pad, pad, pad, pad))])

        if symmetry_x == True:
            x = np.linspace(-pad, Nx + pad - 1, Nx + 2 * pad) * delta
        else:
            x = np.linspace(-Nx // 2 - pad, Nx // 2 +
                            pad - 1, Nx + 2 * pad) * delta
        z = np.linspace(-Nz // 2 - pad, Nz // 2 +
                        pad - 1, Nz + 2 * pad) * delta

        # Coordinate set-up
        zz, xx = np.meshgrid(z, x)
        self.Nx = zz.shape[0]
        self.Nz = zz.shape[1]

        # incident electric and magnetic fields definition on the Yee grid
        fast = np.exp(1j * (k * zz))
        fast_z = np.exp(1j * (k * (zz + delta / 2)))
        zeros = ops.Zeros()
        self.fast = zeros((1, 2, fast.shape[0], fast.shape[1]), ms.float32)
        self.fast[0, 0, :, :] = Tensor.from_numpy(np.real(fast))
        self.fast[0, 1, :, :] = Tensor.from_numpy(np.imag(fast))

        self.fast_z = zeros((1, 2, fast_z.shape[0], fast_z.shape[1]), ms.float32)
        self.fast_z[0, 0, :, :] = Tensor.from_numpy(np.real(fast_z))
        self.fast_z[0, 1, :, :] = Tensor.from_numpy(np.imag(fast_z))

        # perfectly-matched-layer set up
        m = 4
        const = 5
        rx_p = 1 + 1j * const * (xx - x[-1] + pml_thickness * delta) ** m
        rx_p[0:-pml_thickness, :] = 0
        rx_n = 1 + 1j * const * (xx - x[0] - pml_thickness * delta) ** m
        rx_n[pml_thickness::, :] = 0
        rx = rx_p + rx_n
        if symmetry_x == True:
            rx[0:-pml_thickness:, :] = 1
        else:
            rx[pml_thickness:-pml_thickness, :] = 1

        rz_p = 1 + 1j * const * (zz - z[-1] + pml_thickness * delta) ** m
        rz_p[:, 0:-pml_thickness] = 0
        rz_n = 1 + 1j * const * (zz - z[0] - pml_thickness * delta) ** m
        rz_n[:, pml_thickness::] = 0
        rz = rz_p + rz_n
        rz[:, pml_thickness:-pml_thickness] = 1

        rx_inverse = 1 / rx
        rz_inverse = 1 / rz

        self.rx_inverse = zeros((1, 2, rx_inverse.shape[0], rx_inverse.shape[1]), ms.float32)
        self.rx_inverse[0, 0, :, :] = Tensor.from_numpy(np.real(rx_inverse))
        self.rx_inverse[0, 1, :, :] = Tensor.from_numpy(np.imag(rx_inverse))

        self.rz_inverse = zeros((1, 2, rz_inverse.shape[0], rz_inverse.shape[1]), ms.float32)
        self.rz_inverse[0, 0, :, :] = Tensor.from_numpy(np.real(rz_inverse))
        self.rz_inverse[0, 1, :, :] = Tensor.from_numpy(np.imag(rz_inverse))

        # Gradient and laplacian kernels set up
        self.gradient_h_z = zeros(
            (2, 1, 1, 3), ms.float32)
        self.gradient_h_z[:, :, 0, :] = ms.Tensor(
            [-1 / delta, +1 / delta, 0])
        self.gradient_h_x = zeros(
            (2, 1, 3, 1), ms.float32)
        self.gradient_h_x = transpose(self.gradient_h_z, (0, 1, 3, 2))
        self.gradient_h_z_ho = zeros(
            (2, 1, 1, 5), ms.float32)
        self.gradient_h_z_ho[:, :, 0, :] = ms.Tensor(
            [1 / 24 / delta, -9 / 8 / delta, +9 / 8 / delta, -1 / 24 / delta, 0])
        self.gradient_h_x_ho = zeros(
            (2, 1, 5, 1), ms.float32)
        self.gradient_h_x_ho = transpose(self.gradient_h_z_ho, (0, 1, 3, 2))

        self.gradient_e_z = zeros(
            (2, 1, 1, 3), ms.float32)
        self.gradient_e_z[:, :, 0, :] = ms.Tensor(
            [0, -1 / delta, +1 / delta])
        self.gradient_e_x = zeros(
            (2, 1, 3, 1), ms.float32)
        self.gradient_e_x = transpose(self.gradient_e_z, (0, 1, 3, 2))
        self.gradient_e_z_ho = zeros(
            (2, 1, 1, 5), ms.float32)
        self.gradient_e_z_ho[:, :, 0, :] = ms.Tensor(
            [0, 1 / 24 / delta, -9 / 8 / delta, +9 / 8 / delta, -1 / 24 / delta])
        self.gradient_e_x_ho = zeros(
            (2, 1, 5, 1), ms.float32)
        self.gradient_e_x_ho = transpose(self.gradient_e_z_ho, (0, 1, 3, 2))

        self.dd_z_fast = zeros(
            (1, 2, Nx, Nz), ms.float32)
        self.dd_z_fast = self.dd_z(self.fast)[:, :, self.pad:-self.pad:, :]
        self.dd_z_ho_fast = zeros(
            (1, 2, Nx, Nz), ms.float32)
        self.dd_z_ho_fast = self.dd_z_ho(
            self.fast)[:, :, self.pad:-self.pad:, :]

    def construct(self, scat_pot, ri_value):
        diff_x, diff_z = 0, 0
        total, diff = 0, 0
        if self.mode == 'te':
            epsillon = scat_pot * \
                (ri_value ** 2)
            epsillon = ms.numpy.where(epsillon > 1.0, epsillon, ms.Tensor(
                [1], dtype=ms.float32))
            x = self.model(scat_pot)
            total = ops.concat((x[:, 0:1, :, :] + 1, x[:, 1:2, :, :]), 1)

            ey = self.complex_multiplication(total[:, 0:2, :, :],
                                             self.fast[:, :, self.pad:-self.pad:, self.pad:-self.pad:])
            ey_i = self.fast
            ey_s = ey - ey_i[:, :, self.pad:-self.pad:, self.pad:-self.pad:]

            if self.symmetry_x == True:
                ey_s = self.padding_ref(ey_s)
            else:
                ey_s = self.padding_zero(ey_s)

            if self.high_order == 'second':
                diff = self.dd_x_pml(ey_s)[:, :, :, self.pad:-self.pad] \
                    + self.dd_z_pml(ey_s)[:, :, self.pad:-self.pad, :] \
                    + self.dd_z_fast \
                    + self.k ** 2 * (epsillon * ey)

            elif self.high_order == 'fourth':
                diff = self.dd_x_ho_pml(ey_s)[:, :, :, self.pad:-self.pad] \
                    + self.dd_z_ho_pml(ey_s)[:, :, self.pad:-self.pad, :] \
                    + self.dd_z_ho_fast \
                    + self.k ** 2 * (epsillon * ey)

        elif self.mode == 'tm':
            epsillon = scat_pot * \
                (ri_value ** 2)
            epsillon_x = ms.numpy.where(epsillon[:, 0:1, :, :] > 1.0, epsillon[:, 0:1, :, :],
                                     ms.Tensor([1], dtype=ms.float32))
            epsillon_z = ms.numpy.where(epsillon[:, 1:2, :, :] > 1.0, epsillon[:, 1:2, :, :],
                                     ms.Tensor([1], dtype=ms.float32))

            x = self.model(scat_pot)
            total = ops.concat((x[:, 0:1, :, :] + 1, x[:, 1:4, :, :]), 1)

            ex = self.complex_multiplication(
                total[:, 0:2, :, :], self.fast[:, :, self.pad:-self.pad:, self.pad:-self.pad:])
            ex_i = self.fast
            ex_s = ex - ex_i[:, :, self.pad:-self.pad:, self.pad:-self.pad:]

            ez_s = self.complex_multiplication(
                total[:, 2:4, :, :], self.fast_z[:, :, self.pad:-self.pad:, self.pad:-self.pad:])

            if self.symmetry_x == True:
                ex_s = self.padding_zero(ex_s)
                ez_s = self.padding_ref(ez_s)
                ex_s[:, :, 0:self.pad, :] = ms.numpy.flip(
                    ex_s[:, :, self.pad:2 * self.pad, :], [2])
                ez_s[:, :, 0:self.pad, :] = -ez_s[:, :, 0:self.pad, :]
            else:
                ex_s = self.padding_zero(ex_s)
                ez_s = self.padding_zero(ez_s)

            if self.high_order == 'second':
                diff_x = self.dd_z_pml(ex_s)[:, :, self.pad:-self.pad:, :] \
                    + self.dd_z_fast \
                    - self.dd_zx(ez_s)[:, :, self.pad // 2:-self.pad // 2:, self.pad // 2:-self.pad // 2] \
                    + self.k ** 2 * (epsillon_x * ex) \

                diff_z = self.dd_x_pml(ez_s)[:, :, :, self.pad:-self.pad] \
                    - self.dd_xz(ex_s)[:, :, self.pad // 2:-self.pad // 2:, self.pad // 2:-self.pad // 2] \
                    + self.k ** 2 * (epsillon_z * ez_s) \

            elif self.high_order == 'fourth':
                diff_x = self.dd_z_ho_pml(ex_s)[:, :, self.pad:-self.pad:, :] \
                    + self.dd_z_ho_fast \
                    - self.dd_zx_ho_pml(ez_s)[:, :, self.pad//2:-self.pad//2:, self.pad//2:-self.pad//2] \
                    + self.k ** 2 * (epsillon_x * ex)

                diff_z = self.dd_x_ho_pml(ez_s)[:, :, :, self.pad:-self.pad] \
                    - self.dd_xz_ho_pml(ex_s)[:, :, self.pad//2:-self.pad//2:, self.pad//2:-self.pad//2] \
                    + self.k ** 2 * \
                    (epsillon_z * ez_s[:, :, self.pad:-
                     self.pad:, self.pad:-self.pad:])

            diff = ops.concat((diff_x, diff_z), 1)

        return diff, total

    def complex_multiplication(self, a, b):
        # print(a.shape, b.shape)
        r_p = ops.mul(a[:, 0:1, :, :], b[:, 0:1, :, :]) - \
            ops.mul(a[:, 1:2, :, :], b[:, 1:2, :, :])
        i_p = ops.mul(a[:, 0:1, :, :], b[:, 1:2, :, :]) + \
            ops.mul(a[:, 1:2, :, :], b[:, 0:1, :, :])
        return ops.concat((r_p, i_p), 1)

    def complex_conjugate(self, a):
        return ops.concat((-a[:, 1:2, :, :], a[:, 0:1, :, :]), 1)

    def d_e_x(self, x):
        return ops.conv2d(x, self.gradient_e_x, padding=0, group=2)

    def d_e_x_ho(self, x):
        return ops.conv2d(x, self.gradient_e_x_ho, padding=0, group=2)

    def d_h_x(self, x):
        return ops.conv2d(x, self.gradient_h_x, padding=0, group=2)

    def d_h_x_ho(self, x):
        return ops.conv2d(x, self.gradient_h_x_ho, padding=0, group=2)

    def d_e_z(self, x):
        return ops.conv2d(x, self.gradient_e_z, padding=0, group=2)

    def d_e_z_ho(self, x):
        return ops.conv2d(x, self.gradient_e_z_ho, padding=0, group=2)

    def d_h_z(self, x):
        return ops.conv2d(x, self.gradient_h_z, padding=0, group=2)

    def d_h_z_ho(self, x):
        return ops.conv2d(x, self.gradient_h_z_ho, padding=0, group=2)

    def dd_x(self, x):
        return self.d_h_x(self.d_e_x(x))

    def dd_x_ho(self, x):
        return self.d_h_x_ho(self.d_e_x_ho(x))

    def dd_x_pml(self, x):
        return self.complex_multiplication(self.rx_inverse[:, :, 2:-2, :], self.d_h_x(
            self.complex_multiplication(self.rx_inverse[:, :, 1:-1, :], self.d_e_x(x))))

    def dd_x_ho_pml(self, x):
        return self.complex_multiplication(self.rx_inverse[:, :, 4:-4, :], self.d_h_x_ho(
            self.complex_multiplication(self.rx_inverse[:, :, 2:-2, :], self.d_e_x_ho(x))))

    def dd_z(self, x):
        return self.d_h_z(self.d_e_z(x))

    def dd_z_ho(self, x):
        return self.d_h_z_ho(self.d_e_z_ho(x))

    def dd_z_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, :, 2:-2], self.d_h_z(
            self.complex_multiplication(self.rz_inverse[:, :, :, 1:-1], self.d_e_z(x))))

    def dd_z_ho_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, :, 4:-4], self.d_h_z_ho(
            self.complex_multiplication(self.rz_inverse[:, :, :, 2:-2], self.d_e_z_ho(x))))

    def dd_zx(self, x):
        return self.d_h_z(self.d_e_x(x))

    def dd_zx_ho(self, x):
        return self.d_h_z_ho(self.d_e_x_ho(x))

    def dd_zx_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, 1:-1, 1:-1], self.d_h_z(
            self.complex_multiplication(self.rx_inverse[:, :, 1:-1, :], self.d_e_x(x))))

    def dd_zx_ho_pml(self, x):
        return self.complex_multiplication(self.rz_inverse[:, :, 2:-2, 2:-2], self.d_h_z_ho(
            self.complex_multiplication(self.rx_inverse[:, :, 2:-2, :], self.d_e_x_ho(x))))

    def dd_xz(self, x):
        return self.d_h_x(self.d_e_z(x))

    def dd_xz_ho(self, x):
        return self.d_h_x_ho(self.d_e_z_ho(x))

    def dd_xz_pml(self, x):
        return self.complex_multiplication(self.rx_inverse[:, :, 1:-1, 1:-1], self.d_h_x(
            self.complex_multiplication(self.rz_inverse[:, :, :, 1:-1], self.d_e_z(x))))

    def dd_xz_ho_pml(self, x):
        return self.complex_multiplication(self.rx_inverse[:, :, 2:-2, 2:-2], self.d_h_x_ho(
            self.complex_multiplication(self.rz_inverse[:, :, :, 2:-2], self.d_e_z_ho(x))))
