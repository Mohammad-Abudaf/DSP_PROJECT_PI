import random
import time

import numpy as np
import scipy.signal as signal
from skimage.color import rgb2gray
from skimage.transform import rescale
import cv2

from lib.FourierTransform import FourierTransform
from lib.imgfilters import *

from lib.ImageProcessing import ImageProcessing

from matplotlib import pyplot as plt
from matplotlib.image import imread
import librosa as rosa
import sounddevice as sd
import soundfile as sf
from lib.filters import *

class Examples:
    @staticmethod
    def run_raspberrypi():
        vid = cv2.VideoCapture(0)
        while True:
            ret, frame = vid.read()
            # r_scaled = rescale(frame[:, :, 0], 0.70)
            # g_scaled = rescale(frame[:, :, 1], 0.70)
            # b_scaled = rescale(frame[:, :, 2], 0.70)
            # my_dog_scaled = np.stack([r_scaled, g_scaled, b_scaled], axis=2)
            # conv_im1 = ImageProcessing.rgb_convolve2d(my_dog_scaled , identity)
            my_dog_gray = rescale(rgb2gray(frame), 1)
            conv_im1 = signal.convolve2d(my_dog_gray, kernel2[::-1, ::-1]).clip(0, 1)
            cv2.imshow('frame', conv_im1)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        vid.release()
        cv2.destroyAllWindows()

    @staticmethod
    def run_img_compression(img_path, ration):
        original_img = imread(img_path)
        compressed_img = ImageProcessing.compress(original_img, ration)
        # plot the results
        fig, axis = plt.subplots(1, 2)
        axis[0].imshow(original_img)
        axis[1].imshow(compressed_img, cmap="gray")
        plt.show()

    @staticmethod
    def test_dft(vector):
        t0 = time.time()
        results = FourierTransform.direct_DFT(vector)
        print("delta time = ", time.time() - t0)
        print(results)

    @staticmethod
    def audioCompression(audio_path, ratio):
        audio_stereo, sr = rosa.load(audio_path, mono=True)
        audio = np.array(audio_stereo)
        Fs = sr

        ch1 = np.array(audio)
        ch1_length = len(ch1)

        ch1_Fourier = np.fft.fft(ch1)  # performing Fast Fourier Transform
        abs_ch1_Fourier = np.absolute(ch1_Fourier[:ch1_length // 2])  # the spectrum

        plt.plot(np.linspace(0, Fs / 2, ch1_length // 2), abs_ch1_Fourier)
        plt.ylabel('Spectrum')
        plt.xlabel('$f$ (Hz)')
        plt.show()

        eps = ratio
        # Boolean array where each value indicates whether we keep the corresponding frequency
        frequenciesToRemove1 = (1 - eps) * np.sum(abs_ch1_Fourier) < np.cumsum(abs_ch1_Fourier)

        # The frequency for which we cut the spectrum
        f0 = (len(frequenciesToRemove1) - np.sum(frequenciesToRemove1)) * (Fs / 2) / (ch1_length / 2)

        print("f0 : {} Hz".format(int(f0)))

        plt.axvline(f0, color='r')
        plt.plot(np.linspace(0, Fs / 2, ch1_length // 2), abs_ch1_Fourier)
        plt.ylabel('Spectrum')
        plt.xlabel('$f$ (Hz)')
        plt.show()

        D1 = int(Fs / f0)
        print("Downsampling factor : {}".format(D1))

        new_data1 = ch1[::D1]

        # output = np.array([new_data1])
        # print(output)
        # sf.write("audio_compressed.wav", new_data1, int(Fs / D1), 'PCM_32')
        return new_data1, Fs // f0

    @staticmethod
    def denoise_signal():
        dt = 0.001
        t = np.arange(0, 1, dt)
        f = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t)
        f_clean = f
        f = f + 2.5 * np.random.randn(len(t))

        plt.plot(t, f, color='c', label="Noisy")
        plt.plot(t, f_clean, color='k', label="Clean")
        plt.xlim(t[0], t[-1])
        plt.legend()

        plt.show()

        n = len(t)
        fhat = np.fft.fft(f, n)
        PSD = np.abs(fhat) / n
        freq = (1 / (dt * n)) * np.arange(n)
        L = np.arange(1, np.floor(n / 2), dtype="int")

        fig, axis = plt.subplots(2, 1)
        plt.sca(axis[0])
        plt.plot(t, f, color="c", label="Noisy")
        plt.plot(t, f_clean, color="r", label="clean")
        plt.xlim(t[0], t[-1])

        plt.sca(axis[1])
        plt.plot(freq[L], PSD[L], color="c", label="Noisy")
        plt.xlim(freq[L[0]], freq[L[-1]])

        plt.legend()
        plt.show()

        indices = PSD > 100
        PSD_clean = PSD * indices
        fhat = indices * fhat
        ffilt = np.fft.ifft(fhat)

        fig, axis = plt.subplots(3, 1)
        plt.sca(axis[0])
        plt.plot(t, f, color="c", label="Noisy")
        plt.xlim(t[0], t[-1])

        plt.sca(axis[1])
        plt.plot(t, f_clean, color="r", label="clean")
        plt.xlim(t[0], t[-1])

        plt.sca(axis[2])
        plt.plot(freq[L], PSD[L], color="c", label="Noisy")
        plt.plot(freq[L], PSD_clean[L], color="c", label="Noisy")
        plt.xlim(freq[L[0]], freq[L[-1]])
        plt.show()

    @staticmethod
    def denoise_sound(sound_file):
        audio, sr = rosa.load("assets/om-kalthoum.mp3", mono=True)
        noisy_audio = audio + 0.2 * np.random.randn(len(audio))
        original_freq = np.fft.fft(audio)
        noisy_audio_freq = np.fft.fft(noisy_audio)

        fig, axis = plt.subplots(2, 1)

        plt.sca(axis[0])
        plt.plot(noisy_audio, color="c", label="Noisy")
        plt.plot(audio, color="b", label="clean")
        plt.legend()
        plt.show()

        plt.sca(axis[1])
        plt.plot(noisy_audio_freq[:len(noisy_audio_freq)//2], color="c", label="Noisy")
        plt.plot(original_freq[:len(original_freq)//2], color="b", label="clean")
        plt.legend()
        plt.show()

        clean_audio = Filter.gaussian_filter1d(noisy_audio, 5, 100)
        fig, axis = plt.subplots(2, 1)
        clean_audio_freq = np.fft.fft(clean_audio)

        plt.sca(axis[0])
        plt.plot(clean_audio[:len(clean_audio) // 2], color="c", label="Noisy")
        plt.legend()
        plt.show()

        plt.sca(axis[1])
        plt.plot(clean_audio_freq[:len(clean_audio_freq) // 2], color="b", label="clean")
        plt.legend()
        plt.show()
        sd.play(noisy_audio, sr)
        sd.wait()



