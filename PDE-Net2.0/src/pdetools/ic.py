import numpy as np
from typing import List, Tuple, Union


def _periodic_ic_2d(mesh_size: Tuple[int, int],
                    max_freq: int = 4, shift: float = 2., init_range: float = 2.):
    r"""
    generate initial condition:
        u0(x, y) = init_range * w0(x, y) / max(abs(w0(x, y)) + shift.
        w0(x, y): abs(freqs) <= max_freq.
    """
    x = np.random.randn(mesh_size[1], mesh_size[0])
    freqs = np.fft.fftn(x)
    freqs[max_freq + 1: -max_freq] = 0
    freqs[:, max_freq + 1: -max_freq] = 0
    x = np.fft.ifftn(freqs)
    assert np.linalg.norm(x.imag) < 1e-8
    x = x.real
    x = init_range * x / np.abs(x).max()
    x += np.random.uniform(-shift, shift)
    return x


def periodic_ic_2d(mesh_size: Tuple[int, int], batch_size: int,
                   max_freq: int = 4, shift: float = 2., init_range: float = 2., dtype=np.float32):
    r"""
    :return: (batch_size, 2, mesh_size_y, mesh_size_x), dim1: (Ux, Uy).
    """
    batch_ic = []
    for i in range(batch_size):
        u0 = _periodic_ic_2d(mesh_size, max_freq, shift, init_range)
        v0 = _periodic_ic_2d(mesh_size, max_freq, shift, init_range)
        batch_ic.append(np.stack((u0, v0), axis=0))
    batch_ic = np.stack(batch_ic, axis=0)
    # debug
    batch_ic = np.array(batch_ic, dtype=dtype)
    return batch_ic
