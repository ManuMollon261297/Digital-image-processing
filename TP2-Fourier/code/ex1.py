from scipy import signal
from numpy.fft import fft2, fftshift, ifft2
from numpy import log
import cv2 as cv
import numpy as np

h = np.array([[0, 1/6, 0], [1/6, 1/3, 1/6], [0, 1/6, 0]])
# h = fspecial('disk')               ## LOW pass 
# h = fspecial('unsharp')              ## Hi pass

#freqz2(h)

# construyo una imagen (que es una delta en 0)
# suficientemente grande para poder bien el espectro

N = 512
big = np.zeros((N,N))	#make a big image
big[int(N/2)][int(N/2)] = 1  	#unit impulse

h1 = signal.convolve2d(big, h) # conv with the filter
print(len(h1))
S = fft2(h1) # Spectrum
SM = abs(S) # Modulo
IMd = log(1+abs(SM))/log(10)

cv.imshow(mat = fftshift(SM/SM.max()), winname = 'Spectrum')
cv.waitKey(0)
cv.imshow(mat = fftshift(IMd/IMd.max()), winname = 'Spectrum Log Scale')
cv.waitKey(0)