import numpy as np
import cv2 as cv


def _dummy(asd):
    pass


def run_ej4():

    mat1 = np.zeros((90, 90), dtype=np.uint8)
    mat2 = np.zeros((90, 90), dtype=np.uint8)

    cv.rectangle(img=mat1, pt1=(int(0), int(0)), pt2=mat1.shape, color=63, thickness=cv.FILLED)
    cv.rectangle(img=mat2, pt1=(int(0), int(0)), pt2=mat2.shape, color=223, thickness=cv.FILLED)
    cv.rectangle(img=mat2,  pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=127, thickness=cv.FILLED)

    cv.namedWindow(winname='TrackBars', flags=cv.WINDOW_NORMAL)
    cv.createTrackbar('Luminancia', 'TrackBars', 0, 255, _dummy)

    while True:
        pos_actual = cv.getTrackbarPos(trackbarname='Luminancia', winname='TrackBars')
        cv.rectangle(img=mat1, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=pos_actual, thickness=cv.FILLED)
        cv.rectangle(img=mat2, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=pos_actual, thickness=cv.FILLED)
        new_mat = np.hstack((mat1, mat2))
        cv.imshow(winname='Imagen1', mat=new_mat)
        cv.waitKey(int(1000 / 60))
