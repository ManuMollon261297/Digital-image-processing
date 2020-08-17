import cv2 as cv
import numpy as np


def _new_tonality(new_tone):
    pass


def bach_bands(n_bands, n_instances):

    def run_bach_bands(n_bands=n_bands, n_instances=n_instances):

        if (n_bands-1)*17 > 255:
            raise ValueError("Can't fit " + str(n_bands) + " bands on the picture! Try with a lower number!")

        img = np.zeros((480, 640), dtype=np.uint8)

        curr_tonality = 17
        colors = dict(('gray' + str(i), tuple(curr_tonality * i for _ in range(3))) for i in range(n_bands))

        setters_win = 'Bach ' + str(n_instances)
        cv.namedWindow(winname=setters_win, flags=cv.WINDOW_NORMAL)
        cv.createTrackbar('Tonality'+ str(n_instances), setters_win, curr_tonality, 34, _new_tonality)

        while True:

            for offset, k_v in enumerate(colors.items()):
                k, v = k_v
                width = img.shape[1] / len(colors)
                height = img.shape[0]
                pt1 = (int(offset * width), int(0))
                pt2 = (int((offset+1) * width), int(height))

                cv.rectangle(img=img, pt1=pt1, pt2=pt2, color=v, thickness=cv.FILLED)

            cv.imshow(winname='Bach management' + str(n_instances), mat=img)
            cv.waitKey(int(1000/60))
            curr_tonality = cv.getTrackbarPos(trackbarname='Tonality'+ str(n_instances), winname=setters_win)
            colors = dict(('gray' + str(i), tuple(curr_tonality * i for _ in range(3))) for i in range(n_bands))

    return run_bach_bands