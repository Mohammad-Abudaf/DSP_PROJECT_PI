import matplotlib.pylab
import numpy as np
from scipy import signal
from scipy.signal import convolve2d


# all ou
class Filter:
    @staticmethod
    def median(data, number_of_points):
        return signal.medfilt(data, number_of_points)

    # you must encounter the highest frequency on the signal
    @staticmethod
    def fir_low_pass(data, fc, sampled_freq, num_tabs):
        filter = signal.firwin(num_tabs, fc, nyq=sampled_freq//2)
        return signal.convolve(data, filter, mode="same")

    @staticmethod
    def fir_high_pass(data, fc, sampled_freq, num_tabs):
        filter = signal.firwin(num_tabs, fc, nyq=sampled_freq // 2, pass_zero=False)
        return signal.convolve(data, filter, mode="same")

    @staticmethod
    def fir_band_pass(data, fc1, fc2, sampled_freq, num_tabs):
        filter = signal.firwin(num_tabs, [fc1, fc2], nyq=sampled_freq // 2, pass_zero=False)
        return signal.convolve(data, filter, mode="same")

    @staticmethod
    def gaussian_filter1d(data, size, sigma):
        filter_range = np.linspace(-int(size / 2), int(size / 2), size)
        gaussian_filter = [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / (2 * sigma ** 2)) for x in filter_range]
        return signal.convolve(data, gaussian_filter)