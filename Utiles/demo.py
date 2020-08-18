from bach_file import bach_bands
import threading as thr
from color_management import color_detection
from video_management import see_webcam, see_through_phone
import cv2 as cv
import numpy as np


n_instances = 0
bacher_thread1 = thr.Thread(target=bach_bands(n_bands=10, n_instances=n_instances))
n_instances += 1

bacher_thread1.start()
#
# color_detection(cv.imread(filename='./Capture.PNG'))
#
# see_through_phone()
# see_webcam()