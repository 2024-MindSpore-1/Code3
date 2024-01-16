import math
from packaging import version

import mindspore
import mindspore.nn as nn
import mindspore.numpy as np
import mindspore.ops as ops
from mindspore import Tensor

def compute_amplitude(waveforms, lengths=None, amp_type="avg", scale="linear"):
    """Compute amplitude of a batch of waveforms.

    Arguments
    ---------
    waveform : tensor
        The waveforms used for computing amplitude.
        Shape should be `[time]` or `[batch, time]` or
        `[batch, time, channels]`.
    lengths : tensor
        The lengths of the waveforms excluding the padding.
        Shape should be a single dimension, `[batch]`.
    amp_type : str
        Whether to compute "avg" average or "peak" amplitude.
        Choose between ["avg", "peak"].
    scale : str
        Whether to compute amplitude in "dB" or "linear" scale.
        Choose between ["linear", "dB"].

    Returns
    -------
    The average amplitude of the waveforms.

    Example
    -------
    >>> signal = torch.sin(torch.arange(16000.0)).unsqueeze(0)
    >>> compute_amplitude(signal, signal.size(1))
    tensor([[0.6366]])
    """
    expand_dims = ops.ExpandDims()
    if len(waveforms.shape) == 1:
        waveforms = expand_dims(waveforms, 0)

    # assert amp_type in ["avg", "peak"]
    # assert scale in ["linear", "dB"]

    abs = ops.Abs()
    if amp_type == "avg":
        if lengths is None:
            mean = ops.ReduceMean(keep_dims=True)
            out = mean(abs(waveforms), 1)
        else:
            sum = ops.ReduceSum(keep_dims=True)
            wav_sum = sum(abs(waveforms), 1)
            out = wav_sum / lengths
    # elif amp_type == "peak":
    else:
        # out = torch.max(torch.abs(waveforms), dim=1, keepdim=True)[0]
        argmax = ops.ArgMaxWithValue(axis=1, keepdim=True)
        out = argmax(abs(waveforms))[1]

    if scale == "linear":
        return out
    # elif scale == "dB":
    else:
        # return torch.clamp(20 * torch.log10(out), min=-80)  # clamp zeros
        out = 20 * np.log10(out)
        min_value = Tensor(-80, mindspore.float32)
        # max_value = out.max()
        return ops.clip_by_value(out, min_value, out.max())


def notch_filter(notch_freq, filter_width=101, notch_width=0.05):
    """Returns a notch filter constructed from a high-pass and low-pass filter.

    (from https://tomroelandts.com/articles/
    how-to-create-simple-band-pass-and-band-reject-filters)

    Arguments
    ---------
    notch_freq : float
        frequency to put notch as a fraction of the
        sampling rate / 2. The range of possible inputs is 0 to 1.
    filter_width : int
        Filter width in samples. Longer filters have
        smaller transition bands, but are more inefficient.
    notch_width : float
        Width of the notch, as a fraction of the sampling_rate / 2.

    Example
    -------
    #>>> from speechbrain.dataio.dataio import read_audio
    #>>> signal = read_audio('samples/audio_samples/example1.wav')
    #>>> signal = signal.unsqueeze(0).unsqueeze(2)
    #>>> kernel = notch_filter(0.25)
    #>>> notched_signal = convolve1d(signal, kernel)
    """
    sin = ops.Sin()
    concat = ops.Concat()
    ones = ops.Ones()

    # Check inputs
    # assert 0 < notch_freq <= 1
    # assert filter_width % 2 != 0
    pad = filter_width // 2
    inputs = np.arange(filter_width) - pad

    # Avoid frequencies that are too low
    notch_freq += notch_width

    # Define sinc function, avoiding division by zero
    def sinc(x):
        def _sinc(x):
            return sin(x) / x

        # The zero is at the middle index
        return concat([_sinc(x[:pad]), ones(1, mindspore.float32), _sinc(x[pad + 1:])])

    sum = ops.ReduceSum()
    # Compute a low-pass filter with cutoff frequency notch_freq.
    hlpf = sinc(3 * (notch_freq - notch_width) * inputs)
    hlpf *= np.blackman(filter_width)
    hlpf /= sum(hlpf)

    # Compute a high-pass filter with cutoff frequency notch_freq.
    hhpf = sinc(3 * (notch_freq + notch_width) * inputs)
    hhpf *= np.blackman(filter_width)
    hhpf /= -sum(hhpf)
    hhpf[pad] += 1

    # Adding filters creates notch filter
    return (hlpf + hhpf).view((1, -1, 1))


def convolve1d(
    waveform,
    kernel,
    padding=0,
    pad_type="constant",
    stride=1,
    groups=1,
    use_fft=False,
    rotation_index=0,
):
    """Use torch.nn.functional to perform 1d padding and conv.

    Arguments
    ---------
    waveform : tensor
        The tensor to perform operations on.
    kernel : tensor
        The filter to apply during convolution.
    padding : int or tuple
        The padding (pad_left, pad_right) to apply.
        If an integer is passed instead, this is passed
        to the conv1d function and pad_type is ignored.
    pad_type : str
        The type of padding to use. Passed directly to
        `torch.nn.functional.pad`, see PyTorch documentation
        for available options.
    stride : int
        The number of units to move each time convolution is applied.
        Passed to conv1d. Has no effect if `use_fft` is True.
    groups : int
        This option is passed to `conv1d` to split the input into groups for
        convolution. Input channels should be divisible by the number of groups.
    use_fft : bool
        When `use_fft` is passed `True`, then compute the convolution in the
        spectral domain using complex multiply. This is more efficient on CPU
        when the size of the kernel is large (e.g. reverberation). WARNING:
        Without padding, circular convolution occurs. This makes little
        difference in the case of reverberation, but may make more difference
        with different kernels.
    rotation_index : int
        This option only applies if `use_fft` is true. If so, the kernel is
        rolled by this amount before convolution to shift the output location.

    Returns
    -------
    The convolved waveform.

    Example
    -------
    #>>> from speechbrain.dataio.dataio import read_audio
    #>>> signal = read_audio('samples/audio_samples/example1.wav')
    #>>> signal = signal.unsqueeze(0).unsqueeze(2)
    #>>> kernel = torch.rand(1, 10, 1)
    #>>> signal = convolve1d(signal, kernel, padding=(9, 0))
    """
    # if len(waveform.shape) != 3:
    #     raise ValueError("Convolve1D expects a 3-dimensional tensor")

    # Move time dimension last, which pad and fft and conv expect.
    # waveform = waveform.transpose(2, 1)
    # kernel = kernel.transpose(2, 1)
    waveform = waveform.transpose([0, 2, 1])
    kernel = kernel.transpose([0, 2, 1])

    # Padding can be a tuple (left_pad, right_pad) or an int
    if isinstance(padding, tuple):
        # waveform = torch.nn.functional.pad(
        #     input=waveform, pad=padding, mode=pad_type,
        # )
        padding = ((0, 0), (0, 0), padding)
        waveform = nn.Pad(paddings=padding, mode=pad_type)(waveform)


    # This approach uses FFT, which is more efficient if the kernel is large
    # if use_fft:
    #     print("dataprocess/signal_processing.py use_fft is true, no have torch.rfft")
    #     # raise ValueError("dataprocess/signal_processing.py use_fft is true, no have torch.rfft")
    #
    #     # # Pad kernel to same length as signal, ensuring correct alignment
    #     # zero_length = waveform.shape[-1] - kernel.shape[-1]
    #     #
    #     # # Handle case where signal is shorter
    #     # if zero_length < 0:
    #     #     kernel = kernel[..., :zero_length]
    #     #     zero_length = 0
    #     #
    #     # # Perform rotation to ensure alignment
    #     # zeros = ops.Zeros()
    #     # zeros = zeros((kernel.size(0), kernel.size(1), zero_length))
    #     # after_index = kernel[..., rotation_index:]
    #     # before_index = kernel[..., :rotation_index]
    #     # concat = ops.Concat(axis=-1)
    #     # kernel = concat((after_index, zeros, before_index))
    #     #
    #     # # Multiply in frequency domain to convolve in time domain
    #     # if version.parse(torch.__version__) > version.parse("1.6.0"):
    #     #     import torch.fft as fft
    #     #
    #     #     result = fft.rfft(waveform) * fft.rfft(kernel)
    #     #     convolved = fft.irfft(result, n=waveform.size(-1))
    #     # else:
    #     #     f_signal = torch.rfft(waveform, 1)
    #     #     f_kernel = torch.rfft(kernel, 1)
    #     #     sig_real, sig_imag = f_signal.unbind(-1)
    #     #     ker_real, ker_imag = f_kernel.unbind(-1)
    #     #     f_result = torch.stack(
    #     #         [
    #     #             sig_real * ker_real - sig_imag * ker_imag,
    #     #             sig_real * ker_imag + sig_imag * ker_real,
    #     #         ],
    #     #         dim=-1,
    #     #     )
    #     #     convolved = torch.irfft(
    #     #         f_result, 1, signal_sizes=[waveform.size(-1)]
    #     #     )

    # Use the implementation given by torch, which should be efficient on GPU
    # 缺失算子 torch.nn.functional.conv1d
    # if not use_fft:
        # convolved = torch.nn.functional.conv1d(
        #     input=waveform,
        #     weight=kernel,
        #     stride=stride,
        #     groups=groups,
        #     padding=padding if not isinstance(padding, tuple) else 0,
        # )
    padding = padding if not isinstance(padding, tuple) else 0

    convolved = nn.Conv1d(
        in_channels=waveform.shape[1],
        out_channels=kernel.shape[0],
        kernel_size=kernel.shape[-1],
        weight_init=kernel,
        # weight_init="normal",
        stride=stride,
        group=groups,
        pad_mode="pad",
        padding=padding,
    )(waveform)

    # Return time dimension to the second dimension.
    return convolved.transpose([0, 2, 1])

