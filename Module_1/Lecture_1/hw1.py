import numpy as np
from math import floor

def STFT(signal, window, hop_size):
    N = len(signal)
    W = len(window)
    max_frames = floor((N - W) / hop_size) + 1
    stft_matrix = np.zeros((W, max_frames), dtype="complex")

    for i in range(max_frames):
        cur_signal = signal[i * hop_size: i * hop_size + W] * window
        stft_matrix[:, i] = DFT(cur_signal)

    return stft_matrix


def Melspectogram(signal, window, hop_size, mels_num):
    fft_windows = STFT(signal, window, hop_size)
    magnitude = np.abs(fft_windows)**2
    mel = mel_filter(len(window), mels_num)
    return mel @ magnitude


def DFT(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    exp_coef = np.exp(-2j * np.pi * k * n / N)
    return exp_coef @ signal

    
def mel_filter(window_len, mels_num):
    mels_num = int(mels_num)
    weights = np.zeros((mels_num, int(1 + window_len // 2)))

    fftfreqs = np.fft.rfftfreq(n=window_len)
    mel_f = mel_frequencies(mels_num + 2, 0, 11025)

    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)

    for i in range(mels_num):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (mel_f[2 : mels_num + 2] - mel_f[:mels_num])
    weights *= enorm[:, np.newaxis]

    return weights


def mel_frequencies(mels_num, fmin, fmax):
    min_mel = hz_to_mel(fmin)
    max_mel = hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, mels_num)
    return mel_to_hz(mels)


def hz_to_mel(frequencies):
    return 2595.0 * np.log10(1.0 + frequencies / 700.0)


def mel_to_hz(mels):
    return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
