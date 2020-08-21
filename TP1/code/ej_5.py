import cv2 as cv
import numpy as np
import bicubic
import matplotlib.pyplot as plt


def run_ej5_a(img, show=True, same_size=False):

    if same_size:
        downsampled = cv.resize(src=img[1::4, 1::4], interpolation=cv.INTER_NEAREST, fx=0, fy=0,
                                dsize=img.shape)
    else:
        downsampled = img[1::4, 1::4]

    if show:
        cv.imshow(winname='Ej 5-a', mat=downsampled)
        cv.waitKey(0)

    return downsampled


def run_ej5_b(img, show=True, same_size=False):

    if same_size:
        downsampled_dani = cv.resize(src=img[0::4, 0::4], interpolation=cv.INTER_NEAREST, fx=0, fy=0,
                                     dsize=img.shape)
    else:
        downsampled_dani = img[0::4, 0::4]

    if show:
        cv.imshow(winname='Ej 5-b', mat=downsampled_dani)
        cv.waitKey(0)

    return downsampled_dani


def run_ej5_c(img, show=True, method='manopla'):

    if method == 'manopla':
        mat = [[np.uint8(np.mean(img[i:i + 4, j:j+4])) for j in range(0, len(img), 4)] for i in range(0, len(img), 4)]
        mat = np.array(mat)

        mat = cv.resize(src=mat, dsize=img.shape, fx=0, fy=0, interpolation=cv.INTER_LINEAR)

    else:
        mat = cv.boxFilter(src=img, ksize=(4, 4), ddepth=-1, normalize=True)

    if show:
        cv.imshow(winname='Ej 5-c', mat=mat)
        cv.waitKey(0)

    return mat


def _interpolate(x, x1, x2, x1_cte, x2_cte):
    return (abs(x2 - x) / abs(x2 - x1)) * x1_cte + (abs(x - x1) / abs(x2 - x1)) * x2_cte


def run_ej5_d(img, show=False, method = 'bilineal',scale=4, spectrum = True):
    if method == 'bilineal':
        new_img = np.zeros(tuple(i*scale for i in img.shape), dtype=np.uint8)
        new_img[::scale, ::scale] = img[::]
        new_x_treme, new_y_treme = new_img.shape
        x_treme, y_treme = img.shape

        for x in range(new_x_treme):
            for y in range(new_y_treme):
                y1 = y // scale
                y2 = (y + scale) // scale
                x1 = x // scale
                x2 = (x + scale) // scale

                if (x2 < 0 or x2 >= x_treme) and (y2 < 0 or y2 >= y_treme):
                    new_img[x, y] = _interpolate(y, y1*scale, y2*scale,
                                                 _interpolate(x, x1*scale, x2*scale, img[x1, y1], 0),
                                                 _interpolate(x, x1*scale, x2*scale, 0, 0)
                                                 )
                elif x2 < 0 or x2 >= x_treme:
                    new_img[x, y] = _interpolate(y, y1*scale, y2*scale,
                                                 _interpolate(x, x1*scale, x2*scale, img[x1, y1], 0),
                                                 _interpolate(x, x1*scale, x2*scale, img[x1, y2], 0)
                                                 )

                elif y2 < 0 or y2 >= y_treme:
                    new_img[x, y] = _interpolate(y, y1*scale, y2*scale,
                                                 _interpolate(x, x1*scale, x2*scale, img[x1, y1], img[x2, y1]),
                                                 _interpolate(x, x1*scale, x2*scale, 0, 0)
                                                 )
                else:
                    new_img[x, y] = _interpolate(y, y1*scale, y2*scale,
                                                 _interpolate(x, x1*scale, x2*scale, img[x1, y1], img[x2, y1]),
                                                 _interpolate(x, x1*scale, x2*scale, img[x1, y2], img[x2, y2])
                                                 )
    elif method == 'bicubic':
        new_img = bicubic.biDimInterpol(img, reScaleFactor = scale)
    else:
        print('Method Non Existing')

    if spectrum:

        #WITH NUMPY AND MATPLOTLIB
        """f = np.fft.fft2(new_img)
        Mf = np.abs(f)
        magnitude_spectrum = np.abs(np.fft.fftshift(Mf))
        log_magnitude_spectrum = 20*np.log(magnitude_spectrum)
        plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Log Magnitude Spectrum')
        plt.show()"""
        #WITH NUMPY AND OPENCV
        dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
        plt.imshow(magnitude_spectrum, cmap = 'gray')
        plt.title('Log Magnitude Spectrum')
        
        #cv.imshow(winname='Log Magnitud Spectrum', mat = magnitude_spectrum)

    if show:
        cv.imshow(winname='Resized', mat=new_img)
        plt.show()
        cv.waitKey(0)
    
    return new_img



def run_ej5_e():
    pass