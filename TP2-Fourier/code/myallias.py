import cv2 as cv
import numpy as np
from numpy import pi, cos, sin, log, sign, meshgrid
from numpy.fft import fft2, fftshift


xsize = 512
ysize = 512

alpha1 = -pi/6  # Rotationswinkel 1
alpha2 = pi/6  # Rotationswinkel 2

# f2 = 380     # Frequenz 2
# f1 = 180     # Frequenz 1

f1 = 75     # Frequenz 1
f2 = 75     # Frequenz 2

a1 = 1      # Amplitude 1
a2 = 1      # Amplitude 2
phase1 = 0
phase2 = 0

x = np.arange(xsize)/xsize
y = np.arange(ysize)/ysize
# X represents the X-Y plane coordinates
# Y represents the Y-Z plane coordinates
[X, Y] = meshgrid(x, y)

grid1 = cos(alpha1)*X + sin(alpha1)*Y
grid2 = cos(alpha2)*X + sin(alpha2)*Y
print('GRID1:\n', grid1)

im = np.zeros(shape=(xsize, ysize))
print('im.dtype: ', im.dtype)
# cv.imshow(mat=im, winname='Im1')
# cv.waitKey(0)

im1 = sign(a1*sin(2*pi*f1*grid1 + phase1))
im2 = sign(a2*sin(2*pi*f2*grid2 + phase2))
# cv.imshow(mat=np.hstack((im1, im2)), winname='Im1, Im2')
im = im1+im2
cv.imshow(mat=im, winname='imags')
cv.waitKey(0)

# imshow(im,[-(a1+a2) a1+a2])
# imshow(im)
# surf(im,'FaceColor','interp',...
#    'EdgeColor','none',...
#    'FaceLighting','phong')
# camlight left
# camlight headlight

IM = fft2(im)
IMd = log(1+abs(IM))

cv.imshow(mat=IMd/IMd.max(), winname='IMd')
cv.waitKey(0)

# surf(fftshift(abs(IM)),'FaceColor','interp',...
#    'EdgeColor','none',...
#    'FaceLighting','phong')
# camlight left
# camlight headlight