import numpy as np
from typing import List, Tuple, Union


class PDESolver:
    r""" basic pde solver """
    def __init__(self, max_dt):
        self.max_dt = max_dt

    def step(self, init, dt):
        raise NotImplementedError

    def evolution(self, inputs: np.ndarray, T: float):
        r"""
        :param inputs: start.
        :param T: time for evolution.
        :return: outputs(t + T).
        """
        n = int(np.ceil(T / self.max_dt))
        assert n > 0
        dt = T / n
        u = inputs
        for i in range(n):
            u = self.step(u, dt)
        return u


class Burgers2DSolver(PDESolver):
    r"""
    pde solver for 2d burgers equation
        >> inputs: (batch_size, 2, mesh_size_y, mesh_size_x), dim1: (Ux, Uy).
        >>  or: (batch_size, channel_num, H, W)
    """
    def __init__(self,
                 max_dt: float,
                 mesh_size: Tuple[int, int],
                 mesh_range: Tuple[Tuple[float, float], Tuple[float, float]],
                 viscosity: float,
                 force=None):
        r"""
        :param mesh_size: 2d mesh size for pde solver: (dim0_size, dim1_size).
        :param max_dt: time step.
        :param mesh_range: 2d mesh range: ((dim0_start, dim0_end), (dim1_start, dim1_end)).
        :param viscosity: coefficient of viscosity.
        :param force: external force.
        """
        super().__init__(max_dt)
        self.mesh_size = np.array(mesh_size)
        self.mesh_range = np.array(mesh_range)
        dx0 = (mesh_range[0][1] - mesh_range[0][0]) / mesh_size[0]
        dx1 = (mesh_range[1][1] - mesh_range[1][0]) / mesh_size[1]
        assert abs(dx0 - dx1) < 1e-8
        self.dx = dx0
        self.viscosity = viscosity
        self.force = force

    @staticmethod
    def _periodic_pad(inputs: np.ndarray, pad_len: int = 2, axes: Tuple[int, ...] = (0, )):
        r"""
        :param inputs: (batch_size, 2, mesh_size_y, mesh_size_x)
        :return: (batch_size, 2, pad_len + mesh_size_y + pad_len, pad_len + mesh_size_x + pad_len)
        """
        n = inputs.ndim
        inputs = np.transpose(inputs, range(n-1, -1, -1))
        for axis in axes:
            permute = list(range(n))
            permute[axis] = 0
            permute[0] = axis
            inputs = np.transpose(inputs, permute)
            cat_list = [inputs[slice(-pad_len, None)], inputs, inputs[slice(0, pad_len)]]
            inputs = np.concatenate(cat_list, axis=0)
            inputs = np.transpose(inputs, permute)
        inputs = np.transpose(inputs, range(n - 1, -1, -1))
        return inputs

    def upwind_1d_order2(self, u_pad: np.ndarray, u_1d: np.ndarray, axis: int):
        r"""
        return
        :param u_pad: periodic padded inputs, (batch_size, 2, mesh_size_y, mesh_size_x), dim1: (Ux, Uy).
        :param u_1d: (batch_size, 1, mesh_size_y, mesh_size_x), Ux or Uy.
        :param axis: difference axis: 0, 1.
        :return: dim u: u * u_x; dim v: u * v_x
        """
        assert -2 <= axis < 0
        n = u_pad.ndim
        slice_idx = [slice(None), ] * n
        slice_former1 = slice_idx.copy()
        slice_former2 = slice_idx.copy()
        slice_latter1 = slice_idx.copy()
        slice_latter2 = slice_idx.copy()
        slice_origin = slice_idx.copy()
        slice_origin[axis] = slice(2, -2)
        slice_former1[axis] = slice(1, -3)
        slice_former2[axis] = slice(None, -4)
        slice_latter1[axis] = slice(3, -1)
        slice_latter2[axis] = slice(4, None)

        uu_x = np.minimum(0, u_1d) * ((u_pad[tuple(slice_latter1)] - u_pad[tuple(slice_latter2)]) +
                                      3 * (u_pad[tuple(slice_latter1)] - u_pad[tuple(slice_origin)])) / (2 * self.dx)
        uu_x += np.maximum(0, u_1d) * ((u_pad[tuple(slice_former2)] - u_pad[tuple(slice_former1)]) +
                                       3 * (u_pad[tuple(slice_origin)] - u_pad[tuple(slice_former1)])) / (2 * self.dx)
        return uu_x

    def upwind_2d(self, u):
        pad_len = 2
        u_pad_x = self._periodic_pad(u, pad_len, axes=(0, ))
        u_pad_y = self._periodic_pad(u, pad_len, axes=(1, ))
        u_d0 = u[..., :1, :, :]
        u_d1 = u[..., 1:, :, :]
        uu_x__uv_x = self.upwind_1d_order2(u_pad_x, u_d0, axis=-1)
        vv_y__vu_y = self.upwind_1d_order2(u_pad_y, u_d1, axis=-2)
        return uu_x__uv_x + vv_y__vu_y

    def central_difference(self, u: np.ndarray, axis: int):
        r"""
        :param u: inputs. (batch_size, 2, mesh_size_y, mesh_size_x), dim1: (Ux, Uy).
        :param axis: x: -1, y: -2; pad x: 0, pad y: 1.
        :return: axis = -1: dim0: u_xx, dim1: v_xx; axis = -2: dim0: u_yy, dim1: v_yy.
        """
        assert -2 <= axis <= -1
        n = u.ndim
        pad_len = 1
        u_pad = self._periodic_pad(u, pad_len, axes=(-(axis+1), ))
        slice_idx = [slice(None), ] * n
        slice_former1 = slice_idx.copy()
        slice_latter1 = slice_idx.copy()
        slice_former1[axis] = slice(0, -2)
        slice_latter1[axis] = slice(2, None)
        u_xx = ((u_pad[tuple(slice_latter1)] - u) - (u - u_pad[tuple(slice_former1)])) / self.dx**2
        return u_xx

    def right_hand_items(self, u: np.ndarray):
        r"""
        compute the right hand items of 2d burgers equation:
            -u * u_x + viscosity * u_xx;
            -v * v_x + viscosity * v_xx;

            difference scheme:
            u * u_x, v * u_y; u * v_x, v * v_y: 2nd order upwind.
            u_xx, v_xx: central difference.
         """
        convection = - self.upwind_2d(u)
        viscosity = self.viscosity * (self.central_difference(u, axis=-1) + self.central_difference(u, axis=-2))
        rhi = convection + viscosity
        if self.force is not None:
            rhi += self.force
        return rhi

    def runge_kutta_order4(self, u, dt):
        r""" time difference scheme: 4th order Runge-Kutta. """
        k1 = self.right_hand_items(u)
        k2 = self.right_hand_items(u + dt/2 * k1)
        k3 = self.right_hand_items(u + dt/2 * k2)
        k4 = self.right_hand_items(u + dt * k3)
        rhi = dt / 6 * (k1 + 2*k2 + 2*k3 + k4)
        return rhi

    def step(self, u, dt):
        r""" single step of pde solver """
        u = u + self.runge_kutta_order4(u, dt)
        return u
