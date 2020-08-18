import cv2 as cv
import numpy as np


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


def run_ej5_d():
    pass


def run_ej5_e():
    pass