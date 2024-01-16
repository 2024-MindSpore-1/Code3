import numpy as np
from typing import Tuple, List, Union


class PostProcessor:
    def __init__(self,
                 sample_size: Tuple[int, int],
                 axes: Union[Tuple, List] = (0, 1),
                 start_noise_level: float = 1e-3,
                 end_noise_level: float = 1e-3):
        assert len(sample_size) == len(axes)
        self.sample_size = sample_size
        self.operate_axes = axes
        self.start_noise_level = start_noise_level
        self.end_noise_level = end_noise_level

    def down_sample(self, raw: np.ndarray):
        r"""
        :param raw: (batch_size, 2, mesh_size1, mesh_size0); 0: x, 1: y.
        :return: (batch_size, 2, sample_size1, sample_size0)
        """
        raw_size = raw.shape
        n_dim = raw.ndim
        output = raw.copy()
        for axis in self.operate_axes:
            slices = [slice(None, None), ] * n_dim
            sample_interval = raw_size[-axis - 1] // self.sample_size[axis]
            slices[-axis - 1] = slice(0, None, sample_interval)
            output = output[tuple(slices)]
        return output

    def add_noise(self, start: np.ndarray, end: np.ndarray = None):
        r""" add gaussian noise: noise_level * std * N(0, 1) """
        axes = tuple(-self.operate_axes[i] - 1 for i in range(len(self.operate_axes)))
        std = np.std(start, axis=axes, keepdims=True)
        noise = self.start_noise_level * std * np.random.randn(*start.shape)
        start += noise
        if end is not None:
            end += self.end_noise_level * std * np.random.randn(*end.shape)
            return start, end
        else:
            return start
