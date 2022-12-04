import numpy as np


class Signal:
    def __init__(self, freq, sampling_freq, start=0, end=0):
        self.freq = freq
        self.sampling_rate = sampling_freq
        self.t = np.linspace(start, end, sampling_freq)

    def generate_sin_signal(self):
        return np.sin(2 * np.pi * self.freq * self.t)

    def generate_cos_signal(self):
        return np.cos(2 * np.pi * self.freq * self.t)

    def generate_complex_signal(self):
        return self.generate_cos_signal() + 1j * self.generate_sin_signal()

