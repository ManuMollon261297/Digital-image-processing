import cv2
import numpy as np


def blur_img(img):
    gray_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(src=gray_img, ksize=(5, 5), sigmaX=5, sigmaY=5)
    cv2.imshow(winname='GrayScaled', mat=gray_img)
    cv2.imshow(winname='GrayScaledBlurred', mat=blurred)
    cv2.waitKey(0)


def new_value(asd):
    pass


def shape_detection(img):

    hsv_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2HSV)
    setters_win = 'Color Detection'
    cv2.namedWindow(winname=setters_win, flags=cv2.WINDOW_NORMAL)

    hsv = list('HSV')
    minimums = dict((i + 'Min', 0) for i in hsv)
    maximums = dict((a + 'Max', b) for a, b in zip(hsv, [200, 255, 255]))

    for mi, ma in zip(minimums.items(), maximums.items()):
        min_k, mi_v = mi
        max_k, ma_v = ma
        cv2.createTrackbar(min_k, setters_win, mi_v, ma_v, new_value)
        cv2.createTrackbar(max_k, setters_win, mi_v, ma_v, new_value)

    while True:

        cv2.imshow(winname='Color management', mat=hsv_img)
        cv2.waitKey(int(1000/60))
        lowers = []
        uppers = []

        for mi, ma in zip(minimums.items(), maximums.items()):
            min_k, mi_v = mi
            max_k, ma_v = ma
            lowers.append(cv2.getTrackbarPos(trackbarname=min_k, winname='Color Detection'))
            uppers.append(cv2.getTrackbarPos(trackbarname=max_k, winname='Color Detection'))

        lowers = np.array(lowers)
        uppers = np.array(uppers)

        mask = cv2.inRange(src=img, lowerb=lowers, upperb=uppers)
        filtered_img = cv2.bitwise_and(src1=img, src2=img, mask=mask)
        cv2.imshow(winname='External', mat=filtered_img)


blur_img(cv2.imread(filename='../Capture.PNG'))