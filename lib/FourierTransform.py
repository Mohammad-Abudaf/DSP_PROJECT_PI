import math

import numpy as np


class FourierTransform:
    @staticmethod
    def direct_DFT(signal):
        N = len(signal)
        real_freq = [0] * N
        img_freq = [0] * N
        for k in range(N):
            for n in range(N):
                real_freq[k] += np.real(signal[n]) * np.cos(2 * np.pi * k * n / N) \
                                + np.imag(n) * np.sin(2 * np.pi * n * k / N)
                img_freq[k] += np.real(signal[n]) * np.sin(2 * np.pi * k * n / N) \
                               - np.imag(n) * np.cos(2 * np.pi * n * k / N)

        return np.array(real_freq) - 1j * np.array(img_freq)
