import numpy as np
from scipy import signal
from scipy.signal import iirfilter, freqs

import lib.singals as sig
import lib.filters as fil
from matplotlib import pyplot as plt, style

sig1 = sig.Signal(5, 2001, 0, 1.0)
sig2 = sig.Signal(50, 2001, 0, 1.0)
sig3 = sig.Signal(250, 2001, 0, 1.0)


def main():
    b, a = iirfilter(4, [1, 10], 1, 60, analog=True, ftype='cheby1')
    print(b)
    print(a)
    w, h = freqs(b, a, worN=np.logspace(-1, 2, 1000))

    plt.semilogx(w, abs(h))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude Response')
    plt.grid()
    plt.show()

    return None


if __name__ == '__main__':
    main()
