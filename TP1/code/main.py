import cv2 as cv
import numpy as np
from ej_4 import run_ej4
from ej_5 import run_ej5_a, run_ej5_b, run_ej5_c, run_ej5_d


img = cv.imread(filename='../mono.bmp', flags=cv.IMREAD_GRAYSCALE)

# run_ej4()
# run_ej5_a(img=img, show=False, same_size=True)
a = run_ej5_b(img=img, show=True, same_size=False)
# a = run_ej5_c(img=img, show=False, method=None)
# b = run_ej5_c(img=img, show=False)
# new_mat = np.hstack((a, b))
b = run_ej5_d(a, show=True, method = 'bicubic')
c = cv.resize(src=a, dsize=(256, 256), fx=0, fy=0, interpolation=cv.INTER_CUBIC)

cv.imshow(winname='Averaged', mat=np.hstack((c, b)))
cv.waitKey(0)
