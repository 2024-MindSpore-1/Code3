import mindspore as ms
import numpy as np
import mindspore.numpy as ms_np
from mindspore import nn, ops, Parameter
from .utils import _M2K, periodic_pad_2d
from .loss import moment_loss, symnet_loss, data_loss
from typing import List, Tuple, Union
import sympy


class SymNet(nn.Cell):
    def __init__(self, name: str, hidden_layers: int, channels_num: int,  channels_names: List[str], dtype,
                 init_range: float = 0.001):
        r"""
        :param hidden_layers: num of hidden layers.
        :param channels_num: num of candidate right hand items.
        :param dtype: mindspore.float32 or mindspore.float64.
        :param init_range: initial range of weight and bias: (-init_range, init_range).
        """
        super().__init__()
        self.name = name
        self.hidden_layers = hidden_layers
        self.channels_num = channels_num
        self.dtype = dtype
        self.channels_names = channels_names
        self.init_range = init_range
        # hidden layers
        for add_ch in range(self.hidden_layers):
            linear = nn.Dense(self.channels_num + add_ch, 2)
            linear.weight.set_dtype(self.dtype)
            linear.bias.set_dtype(self.dtype)
            # init
            linear.weight.set_data(ms.numpy.rand(linear.weight.value().shape, dtype=self.dtype) * self.init_range)
            linear.bias.set_data(ms.numpy.rand(linear.bias.value().shape, dtype=self.dtype) * self.init_range)
            self.__setattr__(name='hidden_{}'.format(add_ch), value=linear)
        # output layer
        linear = nn.Dense(self.channels_num + hidden_layers, 1)
        linear.weight.set_dtype(self.dtype)
        linear.bias.set_dtype(self.dtype)
        # init
        linear.weight.set_data(ms.numpy.rand(linear.weight.value().shape, dtype=self.dtype) * self.init_range)
        linear.bias.set_data(ms.numpy.rand(linear.bias.value().shape, dtype=self.dtype) * self.init_range)
        self.__setattr__(name='output', value=linear)

    @staticmethod
    def _simplify(expression: sympy.Expr, coe_threshold: float):
        coe_dict = expression.as_coefficients_dict()
        simplified = 0
        for item, coe in coe_dict.items():
            if abs(coe) > coe_threshold:
                simplified = simplified + item * coe
        return simplified

    def expression(self, coe_threshold: float = 5e-3):
        init_candidates = sympy.symbols(self.channels_names)
        candidates = []
        add_candidates = init_candidates
        for hidden_idx in range(self.hidden_layers):
            candidates = candidates + add_candidates
            hidden_candidates = sympy.Matrix([candidates, ])
            hidden_w = self.name_cells()['hidden_{}'.format(hidden_idx)].weight.value().asnumpy()
            hidden_b = self.name_cells()['hidden_{}'.format(hidden_idx)].bias.value().asnumpy()
            hidden_w = sympy.Matrix(hidden_w)
            hidden_b = sympy.Matrix(hidden_b)
            add_candidates = hidden_w * hidden_candidates.transpose() + hidden_b
            add_candidates = [add_candidates[0] * add_candidates[1]]

        candidates = candidates + add_candidates
        candidates = sympy.Matrix([candidates, ])
        output_w = self.name_cells()['output'].weight.value().asnumpy()
        output_b = self.name_cells()['output'].bias.value().asnumpy()
        output_w = sympy.Matrix(output_w)
        output_b = sympy.Matrix(output_b)
        out = output_w * candidates.transpose() + output_b
        out = out.expand()
        simplified_out = self._simplify(expression=out[0, 0], coe_threshold=coe_threshold)
        return simplified_out

    @staticmethod
    def _dense_forward(dense_layer: nn.Dense, inputs):
        hidden_w = dense_layer.weight.value().copy()
        hidden_b = dense_layer.bias.value().copy()
        hidden_w = ms.ops.transpose(hidden_w, (1, 0))
        combine_outputs = ms.ops.matmul(inputs, hidden_w)
        hidden_b = ms.ops.broadcast_to(hidden_b, combine_outputs.shape)
        combine_outputs = combine_outputs + hidden_b
        return combine_outputs

    def pseudo_forward(self, candidate_inputs: ms.Tensor):
        r"""
        :param candidate_inputs: (batch_size, candidates_num, height, weight)
        :return: (batch_size, 1, height, weight)
        """
        assert candidate_inputs.dtype == self.dtype
        outputs = ms.ops.transpose(candidate_inputs, (0, 2, 3, 1))
        for idx in range(self.hidden_layers):
            combine_outputs = self._dense_forward(dense_layer=self.name_cells()['hidden_{}'.format(idx)],
                                                  inputs=outputs)
            outputs = ms.ops.concat((outputs, combine_outputs[..., :1] * combine_outputs[..., 1:]), axis=-1)

        outputs = self._dense_forward(dense_layer=self.name_cells()['output'], inputs=outputs)
        outputs = ms.ops.transpose(outputs, (0, 3, 1, 2))
        return outputs

    def construct(self, candidate_inputs: ms.Tensor):
        r"""
        :param candidate_inputs: (batch_size, candidates_num, height, weight)
        :return: (batch_size, 1, height, weight)
        """
        assert candidate_inputs.dtype == self.dtype
        outputs = ms.ops.transpose(candidate_inputs, (0, 2, 3, 1))
        for idx in range(self.hidden_layers):
            combine_outputs = self.name_cells()['hidden_{}'.format(idx)](outputs)
            outputs = ms.ops.concat((outputs, combine_outputs[..., :1] * combine_outputs[..., 1:]), axis=-1)
        outputs = self.name_cells()['output'](outputs)
        outputs = ms.ops.transpose(outputs, (0, 3, 1, 2))
        return outputs


class FDConv2D(nn.Cell):
    r"""
    Args:
        dx: (dx, dy)
        kernel_size: (height, width)
        order: (order_y, order_x)
    """
    def __init__(self,
                 name: str,
                 dx: Tuple[float, float],
                 kernel_size: Tuple[int, int],
                 order: Tuple[int, int],
                 dtype):
        super().__init__()
        self.name = name
        self.dx = dx
        self.kernel_size = kernel_size
        self.order = order
        self.dtype = dtype
        self.m2k = _M2K(shape=self.kernel_size, dtype=self.dtype)
        self._init_moment()

    def _init_moment(self):
        raw_moment = ms_np.zeros(self.kernel_size, dtype=self.dtype)
        free_moment = ms_np.zeros(self.kernel_size, dtype=self.dtype)
        mask = ms_np.ones(self.kernel_size, dtype=self.dtype)

        # moment
        raw_moment[self.order] = 1
        free_moment[self.order] = 1
        # (order_x, order_y)
        normal_order = tuple(self.order[i] for i in reversed(range(len(self.order))))
        # scale
        scale = ms_np.math_ops.power(ms.Tensor(self.dx, dtype=self.dtype), ms.Tensor(normal_order, dtype=self.dtype))
        scale = 1 / scale.prod(axis=0, keep_dims=True)
        # mask
        total_order = sum(self.order)
        for order_y in range(total_order + 1):
            for order_x in range(total_order + 1):
                if order_x + order_y <= total_order:
                    mask[order_y, order_x] = 0
        self.raw_moment = raw_moment
        self.mask = mask
        self.scale = scale
        self.free_moment = Parameter(free_moment)
        return

    def kernel(self):
        moment = self.raw_moment + self.mask * self.free_moment
        k = self.m2k(moment)
        return k


class PolyPDENet2D(nn.Cell):
    r"""
     2D PDE-Net 2.0.

     Refers to https://arxiv.org/pdf/1812.04426.pdf

     Args:
         dt (float): time step.
         dx (float or tuple(float, float)): spatial mesh grid cell size, (cell_size_x, cell_size_y).
         max_order (int): The maximum total order of PDE.
         kernel_size (int or tuple(int, int)): (height, width).
         symnet_hidden_num (int): the number of hidden layers of each symbolic net.
         if_upwind (bool): whether use pseudo-upwind scheme.
         dtype: mindspore.float32.
     """
    def __init__(self,
                 dt: float,
                 dx: Union[float, Tuple[float, float]],
                 max_order: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 symnet_hidden_num: int,
                 symnet_init_range: float = 0.001,
                 if_upwind: bool = True,
                 dtype=ms.float32):
        super().__init__()
        if isinstance(dx, float):
            dx = tuple(dx for i in range(2))
        if isinstance(kernel_size, int):
            kernel_size = tuple(kernel_size for i in range(2))
        assert len(kernel_size) == 2 and len(dx) == 2
        self.dt = dt
        self.dx = dx
        self.max_order = max_order
        self.kernel_size = kernel_size
        self.pad_size = ((kernel_size[1] - 1)//2, kernel_size[1] - 1 - (kernel_size[1] - 1)//2,
                         (kernel_size[0] - 1)//2, kernel_size[0] - 1 - (kernel_size[0] - 1)//2)
        self.mesh_ndim = 2
        self.symnet_hidden_num = symnet_hidden_num
        self.symnet_init_range = symnet_init_range
        self.if_upwind = if_upwind
        self.dtype = dtype
        self.candidate_items_num = 0
        self.channel_names = ['u', 'v']
        self.symnet_candidates_names = []
        self._prepare_kernels()
        self._prepare_symnet()

    def set_frozen(self, frozen: bool):
        requires_grad = not frozen
        for total_order in range(self.max_order + 1):
            for order_x in range(total_order + 1):
                order_y = total_order - order_x
                fd_name = 'fd_x{}_y{}'.format(order_x, order_y)
                self.name_cells()[fd_name].free_moment.requires_grad = requires_grad
        return

    def _prepare_kernels(self):
        r"""
        a kernel with sum rules of (order_y, order_x) means an at least (order_x, order_y) differential operator
        """
        for total_order in range(self.max_order + 1):
            for order_x in range(total_order + 1):
                order_y = total_order - order_x
                fd_name = 'fd_x{}_y{}'.format(order_x, order_y)
                fd = FDConv2D(name=fd_name, dx=self.dx, kernel_size=self.kernel_size,
                              order=(order_y, order_x), dtype=self.dtype)
                self.__setattr__(name=fd_name, value=fd)
                self.candidate_items_num += 1
        self.candidate_items_num *= self.mesh_ndim
        return

    def _prepare_symnet(self):
        r"""
        for each mesh_dim, a symbolic net combines the outputs of all candidate kernels,
        with sequence:
            mesh_dim_0:
                fd_x0_y0,
                fd_x0_y1, fd_x1_y0,
                fd_x0_y2, fd_x1_y1, fd_x2_y0,
                ...
                fd_x{0}_y{max_order}, ... , fd_x{max_order}_y{0}.
            mesh_dim_1:
                ...
        """
        for ch_name in self.channel_names:
            for total_order in range(self.max_order + 1):
                for order_x in range(total_order + 1):
                    order_y = total_order - order_x
                    candidate_name = '{}_x{}_y{}'.format(ch_name, order_x, order_y)
                    self.symnet_candidates_names.append(candidate_name)

        for dim in range(self.mesh_ndim):
            symnet_name = 'symnet_dim{}'.format(dim)
            symnet = SymNet(name=symnet_name, hidden_layers=self.symnet_hidden_num,
                            channels_num=self.candidate_items_num, channels_names=self.symnet_candidates_names,
                            dtype=self.dtype, init_range=self.symnet_init_range)
            self.__setattr__(name=symnet_name, value=symnet)
        return

    def show_expression(self, coe_threshold: float = 5e-3):
        for idx in range(len(self.channel_names)):
            ch_name = self.channel_names[idx]
            symnet_name = 'symnet_dim{}'.format(idx)
            expression = self.name_cells()[symnet_name].expression(coe_threshold=coe_threshold)
            print('derivative of {}: '.format(ch_name), expression)
        return

    def show_kernels(self):
        for total_order in range(self.max_order + 1):
            for order_x in range(total_order + 1):
                order_y = total_order - order_x
                fd_name = 'fd_x{}_y{}'.format(order_x, order_y)
                kernel = self.name_cells()[fd_name].kernel()
                print('======================= k_x{}_y{} =======================: '.format(order_x, order_y))
                print(kernel)
                print('=========================================================')
        return

    def fd_params(self) -> List[ms.Parameter]:
        r""" return all moment matrices """
        params = []
        cells = self.name_cells()
        for total_order in range(self.max_order + 1):
            for order_y in range(total_order + 1):
                order_x = total_order - order_y
                fd_name = 'fd_x{}_y{}'.format(order_x, order_y)
                fd = cells[fd_name]
                params += fd.trainable_params()
        return params

    def symnet_params(self) -> List[ms.Parameter]:
        r""" return all symnet params """
        params = []
        cells = self.name_cells()
        for mesh_dim in range(self.mesh_ndim):
            symnet = cells['symnet_dim{}'.format(mesh_dim)]
            params += symnet.trainable_params()
        return params

    @staticmethod
    def flip_2d(kernel, axis: int):
        assert len(kernel.shape) == 2 and 0 <= axis <= 1
        eye = ms_np.eye(*kernel.shape, dtype=kernel.dtype)
        flip_eye = ms_np.flip(eye, axis=(0,))
        eyes = ms.ops.stack((flip_eye, eye), axis=0)
        flip_k = ms.ops.matmul(eyes[axis], kernel)
        flip_k = ms.ops.matmul(flip_k, eyes[1 - axis])
        return flip_k

    def kernels(self):
        r""" return kernel: (output_ch, input_ch, height, width) """
        k00 = self.name_cells()['fd_x0_y0'].kernel().reshape(1, 1, *self.kernel_size) * self.name_cells()['fd_x0_y0'].scale
        k01 = self.name_cells()['fd_x0_y1'].kernel() * self.name_cells()['fd_x0_y1'].scale
        k10 = self.name_cells()['fd_x1_y0'].kernel() * self.name_cells()['fd_x1_y0'].scale
        k01_flip = - self.flip_2d(k01, axis=0)
        k10_flip = - self.flip_2d(k10, axis=1)

        k10 = k10.reshape(1, 1, *self.kernel_size)
        k01 = k01.reshape(1, 1, *self.kernel_size)

        k01_flip = k01_flip.reshape(1, 1, *self.kernel_size)
        k10_flip = k10_flip.reshape(1, 1, *self.kernel_size)

        # k01_flip = - ms_np.flip(k01, axis=(2, ))
        # k10_flip = - ms_np.flip(k10, axis=(3, ))
        rest_k = []
        for total_order in range(2, self.max_order + 1):
            for order_x in range(total_order + 1):
                order_y = total_order - order_x
                fd_cell = self.name_cells()['fd_x{}_y{}'.format(order_x, order_y)]
                kernel = fd_cell.kernel().reshape(1, 1, *self.kernel_size) * fd_cell.scale
                rest_k.append(kernel)
        rest_k = ms.ops.concat(rest_k, axis=0)
        return k00, k01, k10, k01_flip, k10_flip, rest_k

    def right_hand_items_old(self, U: ms.Tensor):
        r"""
        u: (batch_size, 2, Height, width)
        rhi: (batch_size, 2, Height, width)
        """
        assert U.shape[1] == self.mesh_ndim
        k00, k01, k10, k01_flip, k10_flip, rest_k = self.kernels()
        U_split = U.split(axis=1, output_num=self.mesh_ndim)
        # U_split = U.split(axis=1, split_size_or_sections=1)
        U_pad = list(periodic_pad_2d(each_u, self.pad_size) for each_u in U_split)
        U00 = list(ms.ops.conv2d(each_u, k00) for each_u in U_pad)
        U01 = list(ms.ops.conv2d(each_u, k01) for each_u in U_pad)
        U10 = list(ms.ops.conv2d(each_u, k10) for each_u in U_pad)
        U01_flip = list(ms.ops.conv2d(each_u, k01_flip) for each_u in U_pad)
        U10_flip = list(ms.ops.conv2d(each_u, k10_flip) for each_u in U_pad)
        U_rest = list(ms.ops.conv2d(each_u, rest_k) for each_u in U_pad)

        if self.if_upwind:
            """ mindspore 1.10.1 specific: """
            def pseudo_forward_dim0(U_01: ms.Tensor, U_10: ms.Tensor):
                pseudo_u = []
                for idx in range(self.mesh_ndim):
                    pseudo_u = pseudo_u + [U00[idx], U_01[idx], U_10[idx], U_rest[idx]]
                pseudo_u = ms.ops.concat(pseudo_u, axis=1)
                output = self.name_cells()['symnet_dim0'](pseudo_u)
                # output = self.name_cells()['symnet_dim0'].pseudo_forward(pseudo_u)
                return output.sum()

            def pseudo_forward_dim1(U_01: ms.Tensor, U_10: ms.Tensor):
                pseudo_u = []
                for idx in range(self.mesh_ndim):
                    pseudo_u = pseudo_u + [U00[idx], U_01[idx], U_10[idx], U_rest[idx]]
                pseudo_u = ms.ops.concat(pseudo_u, axis=1)
                output = self.name_cells()['symnet_dim1'](pseudo_u)
                # output = self.name_cells()['symnet_dim1'].pseudo_forward(pseudo_u)
                return output.sum()

            for each_dim in range(self.mesh_ndim):
                # tensor_dim = ms.Tensor([each_dim], dtype=ms.int32)
                U01_tmp = [U01[each_dim].copy() for each_dim in range(self.mesh_ndim)]
                U01_tmp = ms.ops.stack(U01_tmp, axis=0)
                U10_tmp = [U10[each_dim].copy() for each_dim in range(self.mesh_ndim)]
                U10_tmp = ms.ops.stack(U10_tmp, axis=0)

                if each_dim == 0:
                    pseudo_grad = ms.ops.grad(pseudo_forward_dim0, grad_position=(0, 1))
                    grads_dim = pseudo_grad(U01_tmp, U10_tmp)
                else:
                    pseudo_grad = ms.ops.grad(pseudo_forward_dim1, grad_position=(0, 1))
                    grads_dim = pseudo_grad(U01_tmp, U10_tmp)

                grads_01 = grads_dim[0][each_dim]
                grads_10 = grads_dim[1][each_dim]
                # print('dim{}, grad 01: '.format(each_dim), ms.ops.abs((grads_01 <= 0) * 1. - (U00[1] > 0) * 1.).sum())
                # print('dim{}, grad 10: '.format(each_dim), ms.ops.abs((grads_10 <= 0) * 1. - (U00[0] > 0) * 1.).sum())
                # print('dim{} grad 01'.format(each_dim), grads_01)
                # print('dim{} grad 10'.format(each_dim), grads_10)
                U01[each_dim] = (grads_01 > 0) * U01[each_dim] + (grads_01 <= 0) * U01_flip[each_dim]
                U10[each_dim] = (grads_10 > 0) * U10[each_dim] + (grads_10 <= 0) * U10_flip[each_dim]

        # symnet
        u_candidates = []
        for mesh_dim in range(self.mesh_ndim):
            u_candidates = u_candidates + [U00[mesh_dim], U01[mesh_dim], U10[mesh_dim], U_rest[mesh_dim]]
        u_candidates = ms.ops.concat(u_candidates, axis=1)

        # print(self.name_cells()['symnet_dim{}'.format(0)].construct(u_candidates))
        rhi = list(self.name_cells()['symnet_dim{}'.format(mesh_dim)](u_candidates) for mesh_dim in range(self.mesh_ndim))
        rhi = ms.ops.concat(rhi, axis=1)
        return rhi

    def right_hand_items(self, U: ms.Tensor):
        r"""
        u: (batch_size, 2, Height, width)
        rhi: (batch_size, 2, Height, width)
        """
        assert U.shape[1] == self.mesh_ndim
        k00, k01, k10, k01_flip, k10_flip, rest_k = self.kernels()
        if ms.version.__version__ == '1.10.1':
            U_split = U.split(axis=1, output_num=self.mesh_ndim)
        elif ms.version.__version__ == '2.0.0':
            U_split = U.split(axis=1, split_size_or_sections=1)
        else:
            raise ReferenceError('only mindspore 1.10.1 or mindspore 2.0.0')
        U_pad = list(periodic_pad_2d(each_u, self.pad_size) for each_u in U_split)
        U00 = list(ms.ops.conv2d(each_u, k00) for each_u in U_pad)
        U01 = list(ms.ops.conv2d(each_u, k01) for each_u in U_pad)
        U10 = list(ms.ops.conv2d(each_u, k10) for each_u in U_pad)
        U01_flip = list(ms.ops.conv2d(each_u, k01_flip) for each_u in U_pad)
        U10_flip = list(ms.ops.conv2d(each_u, k10_flip) for each_u in U_pad)
        U_rest = list(ms.ops.conv2d(each_u, rest_k) for each_u in U_pad)

        if self.if_upwind:
            for each_dim in range(self.mesh_ndim):
                U01_tmp = [U01[each_dim].copy() for each_dim in range(self.mesh_ndim)]
                U01_tmp = ms.ops.stack(U01_tmp, axis=0)

                U10_tmp = [U10[each_dim].copy() for each_dim in range(self.mesh_ndim)]
                U10_tmp = ms.ops.stack(U10_tmp, axis=0)

                def pseudo_forward(U_01: ms.Tensor, U_10: ms.Tensor):
                    pseudo_u = []
                    for idx in range(self.mesh_ndim):
                        pseudo_u = pseudo_u + [U00[idx], U_01[idx], U_10[idx], U_rest[idx]]
                    pseudo_u = ms.ops.concat(pseudo_u, axis=1)
                    output = self.name_cells()['symnet_dim{}'.format(each_dim)].pseudo_forward(pseudo_u)
                    return output.sum()

                pseudo_grad = ms.ops.grad(pseudo_forward, grad_position=(0, 1))
                grads_dim = pseudo_grad(U01_tmp, U10_tmp)

                grads_01 = grads_dim[0][each_dim]
                grads_10 = grads_dim[1][each_dim]
                # print('dim{}, grad 01: '.format(each_dim), ms.ops.abs((grads_01 <= 0) * 1. - (U00[1] > 0) * 1.).sum())
                # print('dim{}, grad 10: '.format(each_dim), ms.ops.abs((grads_10 <= 0) * 1. - (U00[0] > 0) * 1.).sum())
                # print('dim{} grad 01'.format(each_dim), grads_01)
                # print('dim{} grad 10'.format(each_dim), grads_10)
                U01[each_dim] = (grads_01 > 0) * U01[each_dim] + (grads_01 <= 0) * U01_flip[each_dim]
                U10[each_dim] = (grads_10 > 0) * U10[each_dim] + (grads_10 <= 0) * U10_flip[each_dim]

        # symnet
        u_candidates = []
        for mesh_dim in range(self.mesh_ndim):
            u_candidates = u_candidates + [U00[mesh_dim], U01[mesh_dim], U10[mesh_dim], U_rest[mesh_dim]]
        u_candidates = ms.ops.concat(u_candidates, axis=1)
        rhi = list(self.name_cells()['symnet_dim{}'.format(mesh_dim)](u_candidates) for mesh_dim in range(self.mesh_ndim))
        rhi = ms.ops.concat(rhi, axis=1)
        return rhi

    def evolution(self, inputs: ms.Tensor, step_num: int):
        dt = ms.Tensor([self.dt], dtype=self.dtype)
        for i in range(step_num):
            # Euler
            rhi = self.right_hand_items_old(inputs)
            inputs = inputs + rhi * dt
            # runge-kutta
            # k1 = self.right_hand_items(inputs)
            # k2 = self.right_hand_items(inputs + k1 * dt/2)
            # k3 = self.right_hand_items(inputs + k2 * dt/2)
            # k4 = self.right_hand_items(inputs + k3 * dt)
            # inputs = inputs + dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        return inputs

    def construct(self, inputs: ms.Tensor, T: float):
        r"""
        :param inputs: (batch_size, 2, height, width).
        :param T: evolution time.
        :return: (batch_size, 2, height, width)
        """
        step_num = np.round(T / self.dt)
        if np.abs(step_num - T / self.dt) > 1e-5:
            raise ValueError('for the sake of accuracy, please set evolution time: integer * dt.')
        return self.evolution(inputs, int(step_num))


class PDENetWithLoss:
    def __init__(self,
                 pde_net: PolyPDENet2D,
                 moment_loss_threshold: float,
                 symnet_loss_threshold: float,
                 moment_loss_scale: float,
                 symnet_loss_scale: float,
                 step_num: int,
                 regularization: bool):
        super().__init__()
        self.pde_net = pde_net
        self.moment_loss_threshold = moment_loss_threshold
        self.symnet_loss_threshold = symnet_loss_threshold
        self.moment_loss_scale = moment_loss_scale
        self.symnet_loss_scale = symnet_loss_scale
        self.step_num = step_num
        self.regularization = regularization
        self.dt = self.pde_net.dt

    def get_loss(self, batch_trajectory: ms.Tensor):
        r"""
        :param batch_trajectory: TNCHW.

        :return:
        """
        assert self.step_num + 1 <= batch_trajectory.shape[0]
        if self.regularization:
            regularization_scale = 1.
        else:
            regularization_scale = 0.

        current_moment_loss = moment_loss(self.pde_net, self.moment_loss_threshold) * regularization_scale
        current_symnet_loss = symnet_loss(self.pde_net, self.symnet_loss_threshold) * regularization_scale
        current_data_loss = 0
        batch_former = batch_trajectory[0]
        for step in range(1, self.step_num + 1):
            batch_mid = self.pde_net.construct(batch_former, T=self.dt)
            current_data_loss += data_loss(predict=batch_mid, label=batch_trajectory[step], dt=self.dt)
            batch_former = batch_mid
        loss = float(self.step_num) * self.moment_loss_scale * current_moment_loss + \
            float(self.step_num) * self.symnet_loss_scale * current_symnet_loss + current_data_loss
        return loss
