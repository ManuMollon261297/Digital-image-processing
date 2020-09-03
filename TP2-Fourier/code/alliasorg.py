import cv2 as cv
import numpy as np

from numpy import pi, cos, sin, log
from numpy.fft import fft2, fftshift

xsize = 1024
ysize = 1024

alpha1 = pi/4  # Rotationswinkel 1
alpha2 = pi/4  # Rotationswinkel 2
f1 = 180     # Frequenz 1
f2 = 8     # Frequenz 2
a1 = 0      # Amplitude 1
a2 = 1  	   # Amplitude 2
phase1 = pi/2
phase2 = 0

[X,Y] = np.meshgrid(np.arange(xsize)/xsize, np.arange(ysize)/ysize)
print(X)
print(Y)
grid1 = cos(alpha1)*X + sin(alpha1)*Y
grid2 = cos(alpha2)*X + sin(alpha2)*Y
im = a1*sin(2*pi*f1*grid1 + phase1) + a2*sin(2*pi*f2*grid2 + phase2)
print(((im+1)*255/2).astype(np.uint8))
cv.imshow(mat=((im+1)*255/2).astype(np.uint8), winname='IMAG 1')
cv.waitKey(0)

IM = fft2(im)
IMd = log(100+abs(IM))/log(10)
cv.imshow(mat=fftshift(IMd/IMd.max()), winname='IMAG 2')
cv.waitKey(0)

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

X = np.linspace(1, 512)
Y = np.linspace(1,512)
x_ = np.linspace(-512, -1)
y_ = np.linspace(-512, -1)
x = np.concatenate((x_, X))
y = np.concatenate((y_, Y))
fxy = 1/(((x**2) + (y**2) )**(0.5))

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(x, y, fxy, 'gray')
plt.show()



