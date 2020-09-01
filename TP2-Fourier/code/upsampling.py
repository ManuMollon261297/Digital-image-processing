import numpy as np
from numpy.fft import fft2, ifft2, fftshift
import cv2 as cv


def upsample(img, factor):

    # Upsample the square image by a factor of m
    n, m = img.shape

    iup = np.zeros(shape=(factor*n, factor*n))
    # Expand input image
    for i in range(n):
        for j in range(m):
            iup[factor*(i-1)+1, factor*(j-1)+1] = img[i, j]

    # Ideal filter
    n, m = iup.shape
    w = 1 / factor

    # calculate fft and Shift the zero-frequency component to the center of the spectrum.
    f = fftshift(fft2(iup))

    for i in range(n):
        for j in range(m):
            r2 = (i-n//2)**2 + (j-m//2)**2
            if r2 > round(n/2*w)**2:
                f[i, j] = 0

    out = factor * factor * abs(ifft2(fftshift(f)))

    return out
