import numpy as np
import cv2 as cv

#Authors: Manuel Mollón, Tomás Agustín González Orlando, Matías Larroque.

def _dummy(asd):
    pass


def run_ej4():
    """In order to show the effect of the contrast in a picture, this code shows two pictures
    each one presents two rectangles: one central rectangle and one background rectangle. The shade of gray of the
    central rectangle
    is the same in each image, but the background change, so the contrast effect is shown because the central rectangles
    seem to be differents.
    """
    # matrices to draw pictures:
    mat1 = np.zeros((90, 90), dtype=np.uint8)
    mat2 = np.zeros((90, 90), dtype=np.uint8)

    cv.rectangle(img=mat1, pt1=(int(0), int(0)), pt2=mat1.shape, color=63, thickness=cv.FILLED)
    cv.rectangle(img=mat2, pt1=(int(0), int(0)), pt2=mat2.shape, color=223, thickness=cv.FILLED)
    cv.rectangle(img=mat2,  pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=127, thickness=cv.FILLED)

    cv.namedWindow(winname='TrackBars', flags=cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('Luminancia', 'TrackBars', 0, 255, _dummy)

    while True:
        pos_actual = cv.getTrackbarPos(trackbarname='Luminancia', winname='TrackBars')
        cv.rectangle(img=mat1, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=pos_actual, thickness=cv.FILLED)
        cv.rectangle(img=mat2, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=pos_actual, thickness=cv.FILLED)
        new_mat = np.hstack((mat1, mat2))
        cv.imshow(winname='Imagen1', mat=new_mat)
        cv.waitKey(int(1000 / 60))


def _another_way():
    # matrices to draw pictures:
    mat1 = np.zeros((90, 90), dtype=np.uint8)
    mat2 = np.zeros((90, 90), dtype=np.uint8)
    new_mat = np.hstack((mat1, mat2))

    # picture 1:
    cv.rectangle(img=mat1, pt1=(int(0), int(0)), pt2=mat1.shape, color=63, thickness=cv.FILLED)
    cv.rectangle(img=mat1, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=127, thickness=cv.FILLED)
    # picture 2:
    cv.rectangle(img=mat2, pt1=(int(0), int(0)), pt2=mat2.shape, color=223, thickness=cv.FILLED)
    cv.rectangle(img=mat2, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=127, thickness=cv.FILLED)

    # trackBar callback:
    def centerToneChanged(asd):
        pos_actual = cv.getTrackbarPos(trackbarname='Luminancia', winname='TrackBars')
        cv.rectangle(img=mat1, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=pos_actual, thickness=cv.FILLED)
        cv.rectangle(img=mat2, pt1=(int(29), int(29)), pt2=(int(59), int(59)), color=pos_actual, thickness=cv.FILLED)
        new_mat = np.hstack((mat1, mat2))
        cv.imshow(winname='Imagen1', mat=new_mat)

    # create window, trackbar and show two pictures together (joined like an horizontal stack)
    cv.namedWindow(winname='TrackBars', flags=cv.WINDOW_AUTOSIZE)
    cv.createTrackbar('Luminancia', 'TrackBars', 0, 255, centerToneChanged)
    new_mat = np.hstack((mat1, mat2))
    cv.imshow(winname='Imagen1', mat=new_mat)

    # loop
    windowsAreClosed = False
    while (not (windowsAreClosed)):
        cv.waitKey(int(1000 / 60))
        windowsAreClosed = not (bool(cv.getWindowProperty('Imagen1', cv.WND_PROP_VISIBLE)) or bool(
            cv.getWindowProperty('TrackBars', cv.WND_PROP_VISIBLE)))