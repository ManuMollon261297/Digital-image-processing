import cv2 as cv
from numpy.fft import fft2, fftshift, ifft2

def downsampling(I,m,filtered=True):
	# Downsample the square image I by a factor of m
	n, M = I.shape
	# Apply ideal filter
	w = 1/m
	F = fftshift(fft2(I))
	if filtered:
	    for i in range(n):
	        for j in range(M):
	            r2=(i-round(n/2))^2+(j-round(n/2))**2
	            if (r2>round((n/2*w)**2)):
	            	F[i][j] = 0 #ojo w=1/m
	Idown = (ifft2(fftshift(F))).real
	# Now downsample
	Idown = cv.resize(Idown, (int(n/m),int(n/m)), interpolation = cv.INTER_NEAREST)
	return Idown

# Load Image
#image = cv.imread(filename = '../data/images/barbara.png', flags = cv.IMREAD_GRAYSCALE)
#cv.imshow(mat = image, winname = 'Original')
#cv.waitKey(0)
#dImage = downsampling(image,2)
#cv.imshow(mat = dImage, winname = 'Downsampled')
#cv.waitKey(0)