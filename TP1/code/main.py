from bach_file import bach_bands
import threading as thr
from color_management import color_detection
from video_management import see_webcam, see_through_phone
import cv2 as cv
import numpy as np


# n_instances = 0
# bacher_thread1 = thr.Thread(target=bach_bands(n_bands=10, n_instances=n_instances))
# n_instances += 1
#
# bacher_thread1.start()
#
# color_detection(cv.imread(filename='./Capture.PNG'))
#
# see_through_phone()
# see_webcam()

# for i in range(100):
#     print(i)

# mat1 = np.zeros((90, 90), dtype=np.uint8)
# mat2 = np.zeros((90, 90), dtype=np.uint8)
#
# cv.rectangle(img=mat1, pt1=(int(0), int(0)), pt2=mat1.shape, color=63, thickness=cv.FILLED)
# cv.rectangle(img=mat2, pt1=(int(0), int(0)), pt2=mat2.shape, color=223, thickness=cv.FILLED)
# cv.rectangle(img=mat2,  pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=127, thickness=cv.FILLED)


# def dummy(asd):
#     pass


# cv.namedWindow(winname='TrackBars', flags=cv.WINDOW_NORMAL)
# cv.createTrackbar('Luminancia', 'TrackBars', 0, 255, dummy)

# while True:
#     pos_actual = cv.getTrackbarPos(trackbarname='Luminancia', winname='TrackBars')
#     cv.rectangle(img=mat1, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=pos_actual, thickness=cv.FILLED)
    # cv.rectangle(img=mat2, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=pos_actual, thickness=cv.FILLED)
    # new_mat = np.hstack((mat1, mat2))
    # cv.imshow(winname='Imagen1', mat=new_mat)
    # cv.waitKey(int(1000 / 60))


img = cv.imread(filename='../mono.bmp', flags=cv.IMREAD_GRAYSCALE)

downsampled_dani = cv.resize(src=img[0::4, 0::4], interpolation=cv.INTER_NEAREST, fx=0, fy=0,
                             dsize=img.shape)
downsampled_new = cv.resize(src=img[1::4, 1::4], interpolation=cv.INTER_NEAREST, fx=0, fy=0,
                            dsize=img.shape)
print('Img:\n', img)

# mat = []
# for i in range(0, len(img), 4):
#     mat.append([np.uint8(np.mean(img[i:i + 4, j:j+4])) for j in range(0, len(img), 4)])
# mat = np.array(mat)

# mat1 = cv.resize(src=img, fx=0, fy=0, dsize=img.shape, interpolation=cv.INTER_LINEAR)
# mat = cv.resize(src=mat, dsize=img.shape, fx=0, fy=0, interpolation=cv.INTER_LINEAR)
mat = cv.boxFilter(src=img, ksize=(4, 4), ddepth=-1, normalize=True)
# new_mat = np.hstack((img, mat))

# print(downsampled_dani)
# cv.imshow(winname='Hola', mat=new_mat)
# cv.waitKey(0)
cv.imshow(winname='Hola', mat=mat)
cv.waitKey(0)