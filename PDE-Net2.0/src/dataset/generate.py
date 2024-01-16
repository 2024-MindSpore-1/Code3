import numpy as np
import mindspore as ms
from .utils import PostProcessor
from ..pdetools import Burgers2DSolver, periodic_ic_2d


class DataGenerator:
    def __init__(self, config: dict):
        self.mesh_size = config['mesh_size']
        self.mesh_range = config['mesh_range']
        self.max_dt = config['max_dt']
        self.max_freq = config['max_freq']
        self.viscosity = config['viscosity']
        self.init_shift = config['init_shift']
        self.init_range = config['init_range']
        self.sample_mesh_size = config['sample_mesh_size']
        self.start_noise_level = config['start_noise_level']
        self.end_noise_level = config['end_noise_level']

        self.dt = config['dt']

        self.solver = Burgers2DSolver(max_dt=self.max_dt, mesh_size=self.mesh_size, mesh_range=self.mesh_range,
                                      viscosity=self.viscosity)
        self.post_processor = PostProcessor(sample_size=self.sample_mesh_size, axes=(0, 1),
                                            start_noise_level=self.start_noise_level, end_noise_level=self.end_noise_level)

    def generate_data(self, data_num: int, step_num: int, require_raw_data: bool = False, dtype=np.float32):
        r"""
        :return: (data_num, step_num + 1, 2, sample_mesh_size_y, sample_mesh_size_x) >> NTCHW.
        """
        # initial condition
        u0 = periodic_ic_2d(mesh_size=self.mesh_size, batch_size=data_num, max_freq=self.max_freq,
                            shift=self.init_shift, init_range=self.init_range, dtype=dtype)
        # down sample
        u0_sample = self.post_processor.down_sample(u0)
        # add noise
        u0_obs = self.post_processor.add_noise(u0_sample)

        trajectory = [u0_obs, ]
        raw_trajectory = [u0, ]
        u = u0
        print('generating data ...')
        for i in range(step_num):
            u = self.solver.evolution(inputs=u, T=self.dt)
            raw_trajectory.append(u)
            u_sample = self.post_processor.down_sample(u)
            _, u_obs = self.post_processor.add_noise(start=u0_sample, end=u_sample)
            trajectory.append(u_obs)
            if (i + 1) % 20 == 0:
                print('step {} generated.'.format(i + 1))
        print('generated.')
        # N T C H W
        trajectory = np.stack(trajectory, axis=1)
        raw_trajectory = np.stack(raw_trajectory, axis=1)
        if require_raw_data:
            return trajectory, raw_trajectory
        else:
            return trajectory


class Dataset:
    def __init__(self, trajectories: np.ndarray, batch_size: int, dtype, shuffle: bool):
        r"""
        :param trajectories: (data_num, step_num + 1, 2, sample_mesh_size_y, sample_mesh_size_x)
        :param batch_size:
        :param dtype:
        :param shuffle:
        """
        self.trajectories = ms.Tensor(trajectories, dtype=dtype)
        self.data_num = self.trajectories.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

    def fetch(self):
        indices = list(range(self.data_num))
        if self.shuffle:
            np.random.shuffle(indices)
        start = 0
        while start < self.data_num:
            end = start + self.batch_size
            if end > self.data_num:
                end = self.data_num
            batch_indices = indices[start:end]
            yield self.trajectories[batch_indices]
            start = end



