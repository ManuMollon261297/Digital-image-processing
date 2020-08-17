import urllib.request
import cv2 as cv
import numpy as np
import time


def see_webcam():

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print('Error while opening camera')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        gray = cv.cvtColor(src=frame, code=cv.COLOR_BGR2GRAY)
        cv.imshow(winname='WebCam', mat=gray)
        if cv.waitKey(1) == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

def _new_value(asd):
    pass

def see_through_phone():
    URL = "http://192.168.0.27:8080" + '/shot.jpg'

    setters_win = 'Color Detection'
    cv.namedWindow(winname=setters_win, flags=cv.WINDOW_NORMAL)

    hsv = list('HSV')
    minimums = dict((i + 'Min', 0) for i in hsv)
    maximums = dict((a + 'Max', b) for a, b in zip(hsv, [200, 255, 255]))

    for mi, ma in zip(minimums.items(), maximums.items()):
        min_k, mi_v = mi
        max_k, ma_v = ma
        cv.createTrackbar(min_k, setters_win, mi_v, ma_v, _new_value)
        cv.createTrackbar(max_k, setters_win, mi_v, ma_v, _new_value)

    # fourcc = cv.VideoWriter_fourcc(*'DIVX')
    # out = cv.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    while True:
        img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
        # img = cv.imdecode(img_arr, flags=cv.IMREAD_GRAYSCALE )
        img = cv.imdecode(img_arr, flags=cv.IMREAD_COLOR )
        img = cv.resize(src=img, dsize=(640, 480))

        lowers = []
        uppers = []

        for mi, ma in zip(minimums.items(), maximums.items()):
            min_k, mi_v = mi
            max_k, ma_v = ma
            lowers.append(cv.getTrackbarPos(trackbarname=min_k, winname='Color Detection'))
            uppers.append(cv.getTrackbarPos(trackbarname=max_k, winname='Color Detection'))

        lowers = np.array(lowers)
        uppers = np.array(uppers)

        mask = cv.inRange(src=img, lowerb=lowers, upperb=uppers)
        filtered_img = cv.bitwise_and(src1=img, src2=img, mask=mask)
        # out.write(filtered_img)
        cv.imshow(winname='IPWebcam', mat=filtered_img)
        cv.waitKey(int(1000 / 60))
