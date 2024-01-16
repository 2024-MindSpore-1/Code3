import os
import math
import numpy

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import Parameter, Tensor
import mindspore.common.dtype as mstype
from mindspore.ops import constexpr

from dataprocess.signal_processing import convolve1d, notch_filter, compute_amplitude


# @constexpr
# def randint_(drop_count_low, drop_count_high, shape):
#     res = numpy.random.randint(drop_count_low, drop_count_high, shape)
#     # res = tuple(tuple(e.tolist()) for e in res)
#     res = int(res)
#     return res

class Resample(nn.Cell):
    """This class resamples an audio signal using sinc-based interpolation.

    It is a modification of the `resample` function from torchaudio
    (https://pytorch.org/audio/transforms.html#resample)

    Arguments
    ---------
    orig_freq : int
        the sampling frequency of the input signal.
    new_freq : int
        the new sampling frequency after this operation is performed.
    lowpass_filter_width : int
        Controls the sharpness of the filter, larger numbers result in a
        sharper filter, but they are less efficient. Values from 4 to 10 are
        allowed.

    Example
    -------
    # >>> from speechbrain.dataio.dataio import read_audio
    # >>> signal = read_audio('samples/audio_samples/example1.wav')
    # >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    # >>> resampler = Resample(orig_freq=16000, new_freq=8000)
    # >>> resampled = resampler(signal)
    # >>> signal.shape
    # torch.Size([1, 52173])
    # >>> resampled.shape
    torch.Size([1, 26087])
    """

    def __init__(
            self, orig_freq=16000, new_freq=16000, lowpass_filter_width=6,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.lowpass_filter_width = lowpass_filter_width

        # Compute rate for striding
        self._compute_strides()
        # assert self.orig_freq % self.conv_stride == 0
        # assert self.new_freq % self.conv_transpose_stride == 0

        # self.first_indices = Parameter(Tensor(0), requires_grad=False)
        # self.weights = Parameter(Tensor(0))

        self.expand_dims = ops.ExpandDims()
        self.zeros = ops.Zeros()
        self.eye = ops.Eye()
        self.ceil = ops.Ceil()
        self.floor = ops.Floor()
        self.zeros_like = ops.ZerosLike()
        self.less = ops.Less()
        self.cos = ops.Cos()
        self.equal = ops.Equal()
        self.sin = ops.Sin()
        self.abs = ops.Abs()

        if not hasattr(self, "first_indices"):
            self._indices_and_weights()

    def _compute_strides(self):
        """Compute the phases in polyphase filter.

        (almost directly from torchaudio.compliance.kaldi)
        """

        # Compute new unit based on ratio of in/out frequencies
        base_freq = math.gcd(self.orig_freq, self.new_freq)
        input_samples_in_unit = self.orig_freq // base_freq
        self.output_samples = self.new_freq // base_freq

        # Store the appropriate stride based on the new units
        self.conv_stride = input_samples_in_unit
        self.conv_transpose_stride = self.output_samples

    def construct(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # if not hasattr(self, "first_indices"):
        #     self._indices_and_weights()

        # Don't do anything if the frequencies are the same
        if self.orig_freq == self.new_freq:
            return waveforms

        unsqueezed = False
        if len(waveforms.shape) == 2:
            # waveforms = waveforms.unsqueeze(1)
            waveforms = self.expand_dims(waveforms, 1)
            unsqueezed = True
        elif len(waveforms.shape) == 3:
            # waveforms = waveforms.transpose(1, 2)
            waveforms = waveforms.transpose(0, 2, 1)
        # else:
        #     raise ValueError("Input must be 2 or 3 dimensions")

        # Do resampling
        resampled_waveform = self._perform_resample(waveforms)

        if unsqueezed:
            resampled_waveform = resampled_waveform.squeeze(axis=1)
        else:
            # resampled_waveform = resampled_waveform.transpose(1, 2)
            resampled_waveform = resampled_waveform.transpose(0, 2, 1)

        return resampled_waveform

    def _perform_resample(self, waveforms):
        """Resamples the waveform at the new frequency.

        This matches Kaldi's OfflineFeatureTpl ResampleWaveform which uses a
        LinearResample (resample a signal at linearly spaced intervals to
        up/downsample a signal). LinearResample (LR) means that the output
        signal is at linearly spaced intervals (i.e the output signal has a
        frequency of `new_freq`). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        https://ccrma.stanford.edu/~jos/resample/
        Theory_Ideal_Bandlimited_Interpolation.html

        https://github.com/kaldi-asr/kaldi/blob/master/src/feat/resample.h#L56

        Arguments
        ---------
        waveforms : tensor
            The batch of audio signals to resample.

        Returns
        -------
        The waveforms at the new frequency.
        """

        # Compute output size and initialize
        batch_size, num_channels, wave_len = waveforms.shape
        window_size = self.weights.shape[1]
        tot_output_samp = self._output_samples(wave_len)
        # resampled_waveform = torch.zeros(
        #     (batch_size, num_channels, tot_output_samp),
        #     device=waveforms.device,
        # )

        tot_output_samp = int(tot_output_samp)
        resampled_waveform = self.zeros(
            (batch_size, num_channels, tot_output_samp),
            mindspore.float32
        )
        # self.weights = self.weights.to(waveforms.device)

        # Check weights are on correct device
        # if waveforms.device != self.weights.device:
        #     self.weights = self.weights.to(waveforms.device)

        # eye size: (num_channels, num_channels, 1)
        # eye = torch.eye(num_channels, device=waveforms.device).unsqueeze(2)
        eye = self.eye(num_channels, num_channels, mindspore.float32)
        eye = self.expand_dims(eye, 2)

        # Iterate over the phases in the polyphase filter
        for i in range(self.first_indices.shape[0]):
            wave_to_conv = waveforms
            first_index = int(self.first_indices[i])
            # first_index = int(self.first_indices[i][0].item())
            if first_index >= 0:
                # trim the signal as the filter will not be applied
                # before the first_index
                wave_to_conv = wave_to_conv[..., first_index:]

            # pad the right of the signal to allow partial convolutions
            # meaning compute values for partial windows (e.g. end of the
            # window is outside the signal length)
            max_index = (tot_output_samp - 1) // self.output_samples
            end_index = max_index * self.conv_stride + window_size
            current_wave_len = wave_len - first_index
            right_padding = max(0, end_index + 1 - current_wave_len)
            left_padding = max(0, -first_index)
            # right_padding = 0
            # if end_index - current_wave_len + 1 > 0:
            #     right_padding = end_index - current_wave_len + 1
            #
            # left_padding = 0
            # if -first_index > 0:
            #     left_padding = -first_index

            # wave_to_conv = torch.nn.functional.pad(
            #     wave_to_conv, (left_padding, right_padding)
            # )
            wave_to_conv = nn.Pad(paddings=((0, 0), (0, 0), (left_padding, right_padding)))(wave_to_conv)

            # torch.nn.functional.conv1d
            # 缺失算子 torch.nn.functional.conv1d
            # conv_wave = torch.nn.functional.conv1d(
            #     input=wave_to_conv,
            #     weight=self.weights[i].repeat(num_channels, 1, 1),
            #     stride=self.conv_stride,
            #     groups=num_channels,
            # )
            # weight_ = ops.tile(self.weights[i], (num_channels, 1, 1))
            # weight_ = ops.broadcast_to(self.weights[i], (num_channels, 1, -1))
            in_channels = wave_to_conv.shape[1]
            # kernel_size = weight_.shape[2]
            kernel_size = self.weights[i].shape[0]
            conv_wave = nn.Conv1d(
                in_channels, num_channels, kernel_size,
                weight_init=ops.broadcast_to(self.weights[i], (num_channels, 1, -1)),
                # weight_init=weight_,
                stride=self.conv_stride,
                group=num_channels
            )(wave_to_conv)

            # we want conv_wave[:, i] to be at
            # output[:, i + n*conv_transpose_stride]
            # 缺失算子 torch.nn.functional.conv_transpose1d
            # dilated_conv_wave = torch.nn.functional.conv_transpose1d(
            #     conv_wave, eye, stride=self.conv_transpose_stride
            # )

            dilated_conv_wave = nn.Conv1dTranspose(
                num_channels, num_channels, 1,
                weight_init=eye,
                stride=self.conv_transpose_stride
            )(conv_wave)

            # pad dilated_conv_wave so it reaches the output length if needed.
            left_padding = i
            previous_padding = left_padding + dilated_conv_wave.shape[-1]
            right_padding = max(0, tot_output_samp - previous_padding)
            # if tot_output_samp - previous_padding > 0:
            #     right_padding = 0
            # dilated_conv_wave = torch.nn.functional.pad(
            #     dilated_conv_wave, (left_padding, right_padding)
            # )
            dilated_conv_wave = nn.Pad(paddings=((0, 0), (0, 0), (left_padding, right_padding)))(dilated_conv_wave)
            dilated_conv_wave = dilated_conv_wave[..., :tot_output_samp]

            resampled_waveform += dilated_conv_wave

        return resampled_waveform

    def _output_samples(self, input_num_samp):
        """Based on LinearResample::GetNumOutputSamples.

        LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a
        frequency of ``new_freq``). It uses sinc/bandlimited
        interpolation to upsample/downsample the signal.

        (almost directly from torchaudio.compliance.kaldi)

        Arguments
        ---------
        input_num_samp : int
            The number of samples in each example in the batch.

        Returns
        -------
        Number of samples in the output waveform.
        """

        # For exact computation, we measure time in "ticks" of 1.0 / tick_freq,
        # where tick_freq is the least common multiple of samp_in and
        # samp_out.
        samp_in = Tensor(self.orig_freq, mstype.int32)
        samp_out = Tensor(self.new_freq, mstype.int32)


        tick_freq = self.abs(samp_in * samp_out) // np.gcd(samp_in, samp_out)
        ticks_per_input_period = tick_freq // samp_in

        # work out the number of ticks in the time interval
        # [ 0, input_num_samp/samp_in ).
        interval_length = input_num_samp * ticks_per_input_period
        if interval_length <= 0:
            return 0
        ticks_per_output_period = tick_freq // samp_out

        # Get the last output-sample in the closed interval,
        # i.e. replacing [ ) with [ ]. Note: integer division rounds down.
        # See http://en.wikipedia.org/wiki/Interval_(mathematics) for an
        # explanation of the notation.
        last_output_samp = interval_length // ticks_per_output_period

        # We need the last output-sample in the open interval, so if it
        # takes us to the end of the interval exactly, subtract one.
        if last_output_samp * ticks_per_output_period == interval_length:
            last_output_samp -= 1

        # First output-sample index is zero, so the number of output samples
        # is the last output-sample plus one.
        num_output_samp = last_output_samp + 1

        return num_output_samp

    def _indices_and_weights(self):
        """Based on LinearResample::SetIndexesAndWeights

        Retrieves the weights for resampling as well as the indices in which
        they are valid. LinearResample (LR) means that the output signal is at
        linearly spaced intervals (i.e the output signal has a frequency
        of ``new_freq``). It uses sinc/bandlimited interpolation to
        upsample/downsample the signal.

        Returns
        -------
        - the place where each filter should start being applied
        - the filters to be applied to the signal for resampling
        """

        # Lowpass filter frequency depends on smaller of two frequencies
        min_freq = min(self.orig_freq, self.new_freq)
        lowpass_cutoff = 0.99 * 0.5 * min_freq

        assert lowpass_cutoff * 2 <= min_freq
        window_width = self.lowpass_filter_width / (2.0 * lowpass_cutoff)

        assert lowpass_cutoff < min(self.orig_freq, self.new_freq) / 2
        # output_t = torch.arange(
        #     start=0.0, end=self.output_samples, device=waveforms.device,
        # )
        output_t = np.arange(start=0.0, stop=self.output_samples)
        output_t /= self.new_freq
        min_t = output_t - window_width
        max_t = output_t + window_width

        min_input_index = self.ceil(min_t * self.orig_freq)
        max_input_index = self.floor(max_t * self.orig_freq)
        num_indices = max_input_index - min_input_index + 1

        max_weight_width = num_indices.max()
        # j = torch.arange(max_weight_width, device=waveforms.device)
        j = np.arange(max_weight_width.asnumpy())
        input_index = self.expand_dims(min_input_index, 1) + self.expand_dims(j, 0)
        delta_t = (input_index / self.orig_freq) - self.expand_dims(output_t, 1)

        weights = self.zeros_like(delta_t)
        inside_window_indices = self.less(delta_t.abs(), window_width)
        index = np.arange(0, inside_window_indices.shape[1])

        # t_eq_zero_indices = delta_t.eq(0.0)
        t_eq_zero_indices = self.equal(delta_t, 0.0)
        t_not_eq_zero_indices = ~t_eq_zero_indices
        for i in range(inside_window_indices.shape[0]):
            inside_window_indices_ = ops.MaskedSelect()(index, inside_window_indices[i])

            # raised-cosine (Hanning) window with width `window_width`
            weights[i][inside_window_indices_] = 0.5 * (
                1
                + self.cos(
                    2
                    * math.pi
                    * lowpass_cutoff
                    / self.lowpass_filter_width
                    * delta_t[i][inside_window_indices_]
                )
            )

            t_eq_zero_indices_ = ops.MaskedSelect()(index, t_eq_zero_indices[i])
            t_not_eq_zero_indices_ = ops.MaskedSelect()(index, t_not_eq_zero_indices[i])
            # sinc filter function
            weights[i][t_not_eq_zero_indices_] *= self.sin(
                2 * math.pi * lowpass_cutoff * delta_t[i][t_not_eq_zero_indices_]
            ) / (math.pi * delta_t[i][t_not_eq_zero_indices_])

            # limit of the function at t = 0
            if t_eq_zero_indices_.shape[0] > 0:
                weights[i][t_eq_zero_indices_] *= 2 * lowpass_cutoff
        # size (output_samples, max_weight_width)
        weights /= self.orig_freq
        # print("weights5 = ", weights)

        self.first_indices = min_input_index
        self.weights = weights

class SpeedPerturb(nn.Cell):
    """Slightly speed up or slow down an audio signal.

    Resample the audio signal at a rate that is similar to the original rate,
    to achieve a slightly slower or slightly faster signal. This technique is
    outlined in the paper: "Audio Augmentation for Speech Recognition"

    Arguments
    ---------
    orig_freq : int
        The frequency of the original signal.
    speeds : list
        The speeds that the signal should be changed to, as a percentage of the
        original signal (i.e. `speeds` is divided by 100 to get a ratio).
    perturb_prob : float
        The chance that the batch will be speed-
        perturbed. By default, every batch is perturbed.

    Example
    -------
    # >>> from speechbrain.dataio.dataio import read_audio
    # >>> signal = read_audio('samples/audio_samples/example1.wav')
    # >>> perturbator = SpeedPerturb(orig_freq=16000, speeds=[90])
    # >>> clean = signal.unsqueeze(0)
    # >>> perturbed = perturbator(clean)
    # >>> clean.shape
    # torch.Size([1, 52173])
    # >>> perturbed.shape
    # torch.Size([1, 46956])
    """

    def __init__(
            self, orig_freq, speeds=[90, 100, 110], perturb_prob=1.0,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.speeds = speeds
        self.perturb_prob = perturb_prob

        # Initialize index of perturbation
        # self.samp_index = Parameter(Tensor(0, mstype.int32), requires_grad=False)

        # Initialize resamplers
        self.resamplers = []
        for speed in self.speeds:
            config = {
                "orig_freq": self.orig_freq,
                "new_freq": self.orig_freq * speed // 100,
            }
            self.resamplers.append(Resample(**config))

        self.uniform_real = ops.UniformReal()
        self.uniform_int = ops.UniformInt()

    def construct(self, waveform):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't perturb (return early) 1-`perturb_prob` portion of the batches

        # if torch.rand(1) > self.perturb_prob:
        #     return waveform.clone()
        if int(self.uniform_real((1, ))[0]) > self.perturb_prob:
            return waveform.copy()

        # Perform a random perturbation
        # self.samp_index = torch.randint(len(self.speeds), (1,))[0]
        samp_index = int(self.uniform_int((1,), Tensor(0, mstype.int32), Tensor(len(self.speeds), mstype.int32))[0])
        # samp_index = randint_(0, len(self.speeds), (1,))
        perturbed_waveform = self.resamplers[samp_index](waveform)

        return perturbed_waveform


class DropFreq(nn.Cell):
    """This class drops a random frequency from the signal.

    The purpose of this class is to teach models to learn to rely on all parts
    of the signal, not just a few frequency bands.

    Arguments
    ---------
    drop_freq_low : float
        The low end of frequencies that can be dropped,
        as a fraction of the sampling rate / 2.
    drop_freq_high : float
        The high end of frequencies that can be
        dropped, as a fraction of the sampling rate / 2.
    drop_count_low : int
        The low end of number of frequencies that could be dropped.
    drop_count_high : int
        The high end of number of frequencies that could be dropped.
    drop_width : float
        The width of the frequency band to drop, as
        a fraction of the sampling_rate / 2.
    drop_prob : float
        The probability that the batch of signals will  have a frequency
        dropped. By default, every batch has frequencies dropped.

    Example
    # -------
    # >>> from speechbrain.dataio.dataio import read_audio
    # >>> dropper = DropFreq()
    # >>> signal = read_audio('samples/audio_samples/example1.wav')
    # >>> dropped_signal = dropper(signal.unsqueeze(0))
    """

    def __init__(
            self,
            drop_freq_low=1e-14,
            drop_freq_high=1,
            drop_count_low=1,
            drop_count_high=2,
            drop_width=0.05,
            drop_prob=1,
    ):
        super().__init__()
        self.drop_freq_low = drop_freq_low
        self.drop_freq_high = drop_freq_high
        # self.drop_count_low = Tensor(drop_count_low, mstype.int32)
        # self.drop_count_high = Tensor(drop_count_high, mstype.int32)
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_width = drop_width
        self.drop_prob = drop_prob

        self.expand_dims = ops.ExpandDims()
        self.uniform_int = ops.UniformInt()
        self.uniform_real = ops.UniformReal()
        self.zeros = ops.Zeros()

    def construct(self, waveforms):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or `[batch, time, channels]`.
        """

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        dropped_waveform = waveforms.copy()
        if int(self.uniform_real((1,))[0]) > self.drop_prob:
            return dropped_waveform

        # Add channels dimension
        if len(waveforms.shape) == 2:
            dropped_waveform = self.expand_dims(dropped_waveform, -1)

        # Pick number of frequencies to drop
        # drop_count = torch.randint(
        #     low=self.drop_count_low, high=self.drop_count_high + 1, size=(1,),
        # )
        drop_count = int(self.uniform_int((1,), Tensor(self.drop_count_low, mstype.int32), Tensor(self.drop_count_high + 1, mstype.int32))[0])
        # drop_count = randint_(self.drop_count_low, self.drop_count_high + 1, (1,))
        # Pick a frequency to drop
        # torch.rand()支持 torch.rand(drop_count)而 ops.UniformReal()不支持 shape=(0, )
        drop_range = self.drop_freq_high - self.drop_freq_low
        if drop_count == 0:
            drop_frequency = Tensor([], mstype.float32)
        else:
            drop_frequency = (
                self.uniform_real((drop_count, )) * drop_range + self.drop_freq_low
            )
        # drop_frequency = (
        #         self.uniform_real((drop_count,)) * drop_range + self.drop_freq_low
        # )

        # Filter parameters
        filter_length = 101
        pad = filter_length // 2

        # Start with delta function
        # drop_filter = torch.zeros(1, filter_length, 1, device=waveforms.device)
        drop_filter = self.zeros((1, filter_length, 1), mindspore.float32)
        drop_filter[0, pad, 0] = 1

        # Subtract each frequency
        for frequency in drop_frequency:
            notch_kernel = notch_filter(frequency, filter_length, self.drop_width)
            drop_filter = convolve1d(drop_filter, notch_kernel, pad)

        # Apply filter
        dropped_waveform = convolve1d(dropped_waveform, drop_filter, pad)

        # Remove channels dimension if added
        return dropped_waveform.squeeze(-1)


class DropChunk(nn.Cell):
    """This class drops portions of the input signal.

    Using `DropChunk` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.

    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to set the
        signal to zero, in samples.
    drop_length_high : int
        The high end of lengths for which to set the
        signal to zero, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped to zero.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped to zero.
    drop_start : int
        The first index for which dropping will be allowed.
    drop_end : int
        The last index for which dropping will be allowed.
    drop_prob : float
        The probability that the batch of signals will
        have a portion dropped. By default, every batch
        has portions dropped.
    noise_factor : float
        The factor relative to average amplitude of an utterance
        to use for scaling the white noise inserted. 1 keeps
        the average amplitude the same, while 0 inserts all 0's.

    Example
    -------
    # >>> from speechbrain.dataio.dataio import read_audio
    # >>> dropper = DropChunk(drop_start=100, drop_end=200, noise_factor=0.)
    # >>> signal = read_audio('samples/audio_samples/example1.wav')
    # >>> signal = signal.unsqueeze(0) # [batch, time, channels]
    # >>> length = torch.ones(1)
    # >>> dropped_signal = dropper(signal, length)
    # >>> float(dropped_signal[:, 150])
    0.0
    """

    def __init__(
            self,
            drop_length_low=100,
            drop_length_high=1000,
            drop_count_low=1,
            drop_count_high=10,
            drop_start=0,
            drop_end=None,
            drop_prob=1,
            noise_factor=0.0,
    ):
        super().__init__()
        self.drop_length_low = drop_length_low
        self.drop_length_high = drop_length_high
        self.drop_count_low = drop_count_low
        self.drop_count_high = drop_count_high
        self.drop_start = drop_start
        self.drop_end = drop_end
        self.drop_prob = drop_prob
        self.noise_factor = noise_factor

        # Validate low < high
        if drop_length_low > drop_length_high:
            raise ValueError("Low limit must not be more than high limit")
        if drop_count_low > drop_count_high:
            raise ValueError("Low limit must not be more than high limit")

        # Make sure the length doesn't exceed end - start
        if drop_end is not None and drop_end >= 0:
            if drop_start > drop_end:
                raise ValueError("Low limit must not be more than high limit")

            drop_range = drop_end - drop_start
            self.drop_length_low = min(drop_length_low, drop_range)
            self.drop_length_high = min(drop_length_high, drop_range)

        self.uniform_real = ops.UniformReal()
        self.expand_dims = ops.ExpandDims()
        self.uniform_int = ops.UniformInt()
        self.uniform_real = ops.UniformReal()

    def construct(self, waveforms, lengths):
        """
        Arguments
        ---------
        waveforms : tensor
            Shape should be `[batch, time]` or `[batch, time, channels]`.
        lengths : tensor
            Shape should be a single dimension, `[batch]`.

        Returns
        -------
        Tensor of shape `[batch, time]` or
            `[batch, time, channels]`
        """

        # Reading input list
        # lengths = (lengths * waveforms.size(1)).long()
        # lengths = (lengths * waveforms.shape[1]).astype("int64")
        lengths = lengths * waveforms.shape[1]
        lengths = Tensor(lengths, mstype.int64)
        batch_size = waveforms.shape[0]
        dropped_waveform = waveforms.copy()

        # Don't drop (return early) 1-`drop_prob` portion of the batches
        if int(self.uniform_real((1,))[0]) > self.drop_prob:
            return dropped_waveform

        # Store original amplitude for computing white noise amplitude
        # clean_amplitude = compute_amplitude(waveforms, self.expand_dims(lengths, 1))
        clean_amplitude = compute_amplitude(waveforms, self.expand_dims(lengths, -1))

        # Pick a number of times to drop
        # drop_times = torch.randint(
        #     low=self.drop_count_low,
        #     high=self.drop_count_high + 1,
        #     size=(batch_size,),
        # )

        drop_count_low = Tensor(self.drop_count_low, mstype.int32)
        drop_count_high = Tensor(self.drop_count_high + 1, mstype.int32)
        drop_times = self.uniform_int(
            (batch_size,),
            drop_count_low,
            drop_count_high
        )

        # Iterate batch to set mask
        for i in range(batch_size):
            if drop_times[i] == 0:
                continue

            # Pick lengths
            # length = torch.randint(
            #     low=self.drop_length_low,
            #     high=self.drop_length_high + 1,
            #     size=(drop_times[i],),
            # )
            drop_times_i = int(drop_times[i])
            length = self.uniform_int(
                (drop_times_i, ),
                drop_count_low,
                drop_count_high
            )

            # Compute range of starting locations
            start_min = self.drop_start
            if start_min < 0:
                start_min += lengths[i]
            start_max = self.drop_end
            if start_max is None:
                start_max = lengths[i]
            if start_max < 0:
                start_max += lengths[i]

            # start_max = max(0, start_max - length.max())
            if start_max - length.max() > 0:
                start_max = start_max - length.max()
            else:
                start_max = 0

            # Pick starting locations
            # start = torch.randint(
            #     low=start_min, high=start_max + 1, size=(drop_times[i],),
            # )
            start_min = Tensor(start_min, mstype.int32)
            start_max = Tensor(start_max + 1, mstype.int32)
            # start = self.uniform_int(
            #     (drop_times[i],), start_min, start_max + 1
            # )
            start = self.uniform_int(
                (drop_times_i,), start_min, start_max
            )

            end = start + length

            # Update waveform
            if not self.noise_factor:
                for j in range(drop_times[i]):
                    # dropped_waveform[i, int(start[j]): int(end[j])] = 0.0
                    start_j = int(start[j])
                    end_j = int(end[j])
                    dropped_waveform[i, start_j: end_j] = 0.0
            else:
                # Uniform distribution of -2 to +2 * avg amplitude should
                # preserve the average for normalization
                noise_max = 2 * clean_amplitude[i] * self.noise_factor
                for j in range(drop_times[i]):
                    # zero-center the noise distribution
                    # noise_vec = torch.rand(length[j])
                    length_j = int(length[j])
                    start_j = int(start[j])
                    end_j = int(end[j])
                    noise_vec = self.uniform_real((length_j,))
                    noise_vec = 2 * noise_max * noise_vec - noise_max
                    # dropped_waveform[i, int(start[j]): int(end[j])] = noise_vec
                    dropped_waveform[i, start_j: end_j] = noise_vec

        return dropped_waveform

if __name__ == '__main__':
    from mindspore import context
    target = "Ascend"
    device_id = 4

    # init context
    context.set_context(mode=context.GRAPH_MODE, device_target=target, save_graphs=False)
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=target, save_graphs=False)
    context.set_context(device_id=device_id)

    from dataprocess.dataio import read_audio

    signal = read_audio('/old/gxl/sepformer/example1.wav')
    # signal = signal.unsqueeze(0) # [batch, time, channels]
    expand_dims = ops.ExpandDims()
    signal = expand_dims(signal, 0)
    resampler = Resample(orig_freq=16000, new_freq=8000)
    resampled = resampler(signal)
    print(signal.shape)
    # torch.Size([1, 52173])
    print(resampled.shape)
    # torch.Size([1, 26087])


