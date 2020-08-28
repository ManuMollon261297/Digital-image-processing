import cv2 as cv
from numpy import pi, cos, sin, log
from numpy.fft import fft2, fftshift
from scipy.ndimage import affine_transform

f = cv.imread(filename = '../data/images/barbara.png', flags = cv.IMREAD_GRAYSCALE)
(ysize,xsize) = f.shape

mov_pics = cv.VideoWriter('aliasing_pics.avi', apiPreference=cv.CAP_MODE_BGR)
mov_specs = cv.VideoWriter('aliasing_specs.avi', apiPreference=cv.
	)

for xshrink in range(0, 600, 5):
	desiredxsize = xsize - xshrink
	scale_shrink = desiredxsize / xsize
	arrayT1 = np.array([[scale_shrink, 0, 0], [0, scale_shrink, 0], [0, 0, 1]])
	f2 = affine_transform(f,arrayT1)
	(currentysize, currentxsize) = f2.shape

	scale_boost = xsize / currentxsize
	arrayT2 = np.array([[scale_boost, 0, 0], [0, scale_boost, 0], [0, 0, 1]])

	f3 = affine_transform(f2,arrayT2)
	f3 = imtransform(f2,Tinv,'size', output_shape = (ysize, xsize))

	Fd = fftshift(log(1+abs(fft2(f3)))/log(10))

	#fr = im2frame(f3, gray(256)) #CONV
	#Fdr = im2frame(uint8(256*Fd/Fd.max()), gray(256)) #CONV

	mov_pics.write(fr)
	mov_specs.write(Fdr)