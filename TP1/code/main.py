import cv2 as cv
import numpy as np
from ej_4 import run_ej4
from ej_5 import run_ej5_a, run_ej5_b, run_ej5_c, run_ej5_d


img = cv.imread(filename='../mono.bmp', flags=cv.IMREAD_GRAYSCALE)

# run_ej4()
# run_ej5_a(img=img, show=True, same_size=False)
# run_ej5_b(img=img, show=True, same_size=False)
a = run_ej5_c(img=img, show=False, method=None)
b = run_ej5_c(img=img, show=False)
new_mat = np.hstack((a, b))
# run_ej5_d()

cv.imshow(winname='Averaged', mat=new_mat)
cv.waitKey(0)
