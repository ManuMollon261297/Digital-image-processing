import cv2 as cv
import numpy as np
from numpy import pi, cos, sin, log, sign, angle, meshgrid
from numpy.fft import fft2, fftshift


# Prepare image
f = np.zeros(shape=(30, 30))
f[4:23, 12:16] = 1
cv.imshow(mat=f, winname='f')
# Compute Fourier Transform
F = fft2(a=f, s=(256, 256))
F = fftshift(F)     # Center FFT

# Measure the minimum and maximum value of the transform amplitude
np.min(np.min(abs(F)))    # 0
np.max(np.max(abs(F)))    # 100
cv.imshow(mat=abs(F), winname='abs(F)')
cv.waitKey(0)
# colormap(jet)
# colorbar
cv.imshow(mat=log(1+abs(F)), winname='log(1+abs(F))')
cv.waitKey(0)

# colormap(jet)
# colorbar
# What is the main difference between representing the amplitude and its logarithm?
# Look at the phases
# cv.imshow(angle(F), [-pi, pi])
cv.imshow(mat=angle(F), winname='angle(F)')
cv.waitKey(0)

# colormap(jet)
# colorbar
# * Try with other images
# f = imread('saturn.tif')
# f = ind2gray(f,gray(256))
