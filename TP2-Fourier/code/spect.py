import cv2 as cv
import numpy as np
from numpy import pi, cos, sin, log, sign, angle, meshgrid
from numpy.fft import fft2, fftshift


# Prepare image
f = np.zeros(shape=(200, 200))
f[26:153, 80:107] = 1
cv.imshow(mat=f, winname='f')
cv.waitKey(0)

# f = np.zeros(shape=(30, 30))
# f[4:23, 12:16] = 1
# cv.imshow(mat=f, winname='f')
# cv.waitKey(0)
# Compute Fourier Transform
F = fft2(a=f, s=(256, 256))
F = fftshift(F)     # Center FFT

# Measure the minimum and maximum value of the transform amplitude
print(np.min(abs(F)))
print(np.max(abs(F)))
print(np.min(abs(F)))    # 0
print(np.max(log(1+abs(F))))
# cv.imshow(mat=abs(F), winname='abs(F)')
# cv.waitKey(0)
# colormap(jet)
# colorbar
cv.imshow(mat=np.hstack((abs(F) / np.max(abs(F))*255, log(1+abs(F)) / np.max(log(1+abs(F))) *255)), winname='log(1+abs(F))')
cv.waitKey(0)

# colormap(jet)
# colorbar
# What is the main difference between representing the amplitude and its logarithm?
# Look at the phases
# cv.imshow(angle(F), [-pi, pi])
cv.imshow(mat=angle(F), winname='angle(F)')
cv.waitKey(0)

