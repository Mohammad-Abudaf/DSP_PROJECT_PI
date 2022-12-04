import numpy as np
from scipy.signal import convolve2d
from matplotlib.image import imread
from matplotlib.image import imsave
# from lib.FastFourierLib import FastFourier as fft
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

class ImageProcessing:

    @staticmethod
    def rgb_convolve2d(image, kernel):
        red = convolve2d(image[:, :, 0], kernel, 'valid')
        green = convolve2d(image[:, :, 1], kernel, 'valid')
        blue = convolve2d(image[:, :, 2], kernel, 'valid')
        return np.stack([red, green, blue], axis=2)

    @staticmethod
    def compress(img, ratio):
        A = img
        B = np.mean(A, -1)
        Bt = np.fft.fft2(B)
        Btsotred = np.sort(np.abs(Bt.reshape(-1)))
        keep = ratio
        threshold = Btsotred[int(np.floor((1 - keep) * len(Btsotred)))]
        index = np.abs(Bt) > threshold
        AtLow = Bt * index
        Alow = np.fft.ifft2(AtLow).real
        imsave("resultImg.jpg", Alow)
        return Alow

    @staticmethod
    def conv2gray(img, ratio):
        my_dog_gray = rescale(rgb2gray(img), ratio)
        return my_dog_gray
