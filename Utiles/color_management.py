import cv2 as cv
import numpy as np


def new_value(asd):
    pass


def color_detection(img):
    hsv_img = cv.cvtColor(src=img, code=cv.COLOR_BGR2HSV)
    setters_win = 'Color Detection'
    cv.namedWindow(winname=setters_win, flags=cv.WINDOW_NORMAL)

    hsv = list('HSV')
    minimums = dict((i + 'Min', 0) for i in hsv)
    maximums = dict((a + 'Max', b) for a, b in zip(hsv, [200, 255, 255]))

    for mi, ma in zip(minimums.items(), maximums.items()):
        min_k, mi_v = mi
        max_k, ma_v = ma
        cv.createTrackbar(min_k, setters_win, mi_v, ma_v, new_value)
        cv.createTrackbar(max_k, setters_win, mi_v, ma_v, new_value)

    while True:

        cv.imshow(winname='Color management', mat=hsv_img)
        cv.waitKey(int(1000/60))
        lowers = []
        uppers = []

        for (min_k, mi_v), (max_k, ma_v) in zip(minimums.items(), maximums.items()):
            lowers.append(cv.getTrackbarPos(trackbarname=min_k, winname='Color Detection'))
            uppers.append(cv.getTrackbarPos(trackbarname=max_k, winname='Color Detection'))

        lowers = np.array(lowers)
        uppers = np.array(uppers)

        mask = cv.inRange(src=img, lowerb=lowers, upperb=uppers)
        filtered_img = cv.bitwise_and(src1=img, src2=img, mask=mask)
        cv.imshow(winname='External', mat=filtered_img)
