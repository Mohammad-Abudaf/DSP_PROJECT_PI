import numpy as np
import matplotlib.pyplot as plt
import librosa as rosa
import scipy.signal as signal
import lib.filters as filters
import sounddevice as sd

from examples.examples import Examples
def main():
    # sound_stereo, sr = rosa.load("assets/om-kalthoum.mp3", mono=False)
    # channelOne = sound_stereo[0]
    # channelTwo = sound_stereo[1]
    # ch1t = np.fft.fft(channelOne)
    # Mg = np.abs(ch1t)
    # plt.plot(Mg[0:len(Mg) // 2])
    # plt.show()
    # filtered_music = filters.Filter.fir_band_pass(channelOne, 300, 3400, sr, 201)
    # magnitude = np.abs(np.fft.fft(filtered_music))
    # plt.plot(magnitude[0:len(magnitude) // 2])
    # plt.show()
    # sd.play(filtered_music, sr)
    # sd.wait()

    # n = 1024
    # w = np.exp(-1j * 2 * np.pi / n)
    # rows, cols = (n, n)
    # arr = [[0] * cols] * rows
    #
    # for i in range(n):
    #     for j in range(n):
    #         W = np.power(w, i * j)
    #         arr[i][j] = W
    #
    # real = np.real(arr)
    # plt.imshow(real)
    #
    # plt.show()

    Examples.denoise_sound("assets/om-kalthoum-2.wav")


    return None



if __name__ == '__main__':
    main()
