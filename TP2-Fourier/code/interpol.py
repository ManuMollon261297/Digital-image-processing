import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

cap = cv.VideoCapture('barbara.gif')
ret, f1 = cap.read()
cap.release()

f1 = cv.cvtColor(src=f1, code=cv.COLOR_BGR2GRAY)
# f1 = f1[:500, :500]
m = 2   # downsampling factor

# Downsample to a 1/mth of the size, no antialias
f2_aux = cv.resize(src=f1, dsize=(0, 0), fx=1/m, fy=1/m, interpolation=cv.INTER_NEAREST)

# Downsample to a 1/mth of the size, antialias
f2 = cv.resize(src=f1, dsize=(0, 0), fx=1/m, fy=1/m, interpolation=cv.INTER_CUBIC)

# returns a circular averaging filter (pillbox) within the square matrix of size 2*radius+1.
# Pareceria no estar en openCv
# Parece muy abrupto.
# https://www.analog.com/media/en/technical-documentation/dsp-book/dsp_book_Ch24.pdf tiene imagenes.
# Tambien en el codigo se puede ver para muchos casos
# Es muy sharp la caida a mi entender.
# h = fspecial('disk', 1)

# ESTO QUE VIENE ES DISTINTO PERO ESTA COOL!!!
# h = cv.getStructuringElement(shape=cv.MORPH_ELLIPSE, ksize=(5, 5))
# aux = cv.morphologyEx(src=f1, kernel=h, borderType=cv.BORDER_REPLICATE, op=cv.MORPH_DILATE)
# cv.imshow(mat=np.hstack((aux,f1)), winname='asd')
# cv.waitKey(0)
# print(h)


# GaussianBlur mucho mas suave. Este si esta en opencv como parte de los smoothing filters.
# h = fspecial('gaussian',[8 8],.8)
# GaussianBlur no permite tener un kernel de tamaño par al parecer...

gaussianed1 = cv.GaussianBlur(src=f1, sigmaX=0.8, sigmaY=0.8, ksize=(9, 9), borderType=cv.BORDER_DEFAULT)

# el kernel gaussiano es linearmente separable, ver https://theailearner.com/tag/cv2-getgaussiankernel/
# entonces, se puede aplicar primero en x y despues en y el filtro y pasa a ser de orden cuadratico a orden lineal
# el kernel 2D se puede obtener a traves de np.multiply(h,np.transpose(h))
# para aplicar un filtro que es linealmente separable, se puede usar sepFilter2D
h = cv.getGaussianKernel(ksize=9, sigma=0.8, ktype=cv.CV_32F)
gaussianed2 = cv.sepFilter2D(src=f1, ddepth=-1, kernelX=h, kernelY=h)

# Usar sepFilter2D obteniendo el kernel por separado es casi lo mismo que hacer el GaussianBlur.
# Ver el siguiente ejemplo: (menos del 1% de pixeles distintos)
result = cv.sepFilter2D(src=f1, ddepth=-1, kernelX=h, kernelY=h)
result2 = cv.GaussianBlur(src=f1, ksize=(9, 9), sigmaY=1, sigmaX=1, borderType=None)
# diff = result - result2
# print(len(diff[diff != 0]))
# Ademas, el error es muy chico: O dan 1 los pixeles o 255, es decir, un error de un nivel de cuantizacion nomas
# aux = diff[diff >= 2]
# aux = aux[aux != 255]
# print(aux)


# f3 = cv.resize(f2, m)    # Go back to original size
# cv.imshow(mat=f1, winname='f1')
# cv.imshow(mat=f2, winname='f2')
# cv.imshow(mat=np.hstack((f2_aux, f2)), winname='f3')
# f2 = cv.resize(src=f1, dsize=(0, 0), fx=1/2, fy=1/2, interpolation=cv.INTER_AREA)
# kernel_size, sigma = (3, 5)
# result2 = cv.GaussianBlur(src=f2, ksize=(kernel_size, kernel_size), sigmaY=sigma, sigmaX=sigma, borderType=None)

img = np.hstack((f2, result2))
cv.imshow(mat=img, winname='f3')
cv.waitKey(0)
# file_name = 'kernel='+str(kernel_size)+', sigma='+str(sigma) + '.jpg'

# Construct 2D Gaussian Kernel from the linearly separable 1D Gaussian Kernel
kernel = np.multiply(h, np.transpose(h))

# Compute Spectrum
dft = cv.dft(np.float32(kernel), flags=cv.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

# Plot Kernel
plt.subplot(121), plt.imshow(kernel, cmap='gray')
plt.title('Kernel'), plt.xticks([]), plt.yticks([])

# Plot Spectrum
plt.subplot(122), plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()


fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(111, projection='3d')


x = np.arange(kernel.shape[0])
y = np.arange(kernel.shape[1])
X, Y = np.meshgrid(x, y)


mycmap = plt.get_cmap('gist_earth')
ax1.set_title('Gaussian Kernel')
surf1 = ax1.plot_surface(X, Y, kernel, cmap=mycmap)
fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)


plt.show()