import cv2 as cv
import numpy as np
from ej_4 import run_ej4
from ej_5 import run_ej5_a, run_ej5_b, run_ej5_c, run_ej5_d


#ejercicio 3:
"""
img_camera = cv.imread(filename='../fotoIPHONE.jpeg')
print(img_camera.shape)
"""


img = cv.imread(filename='../mono.bmp', flags=cv.IMREAD_GRAYSCALE)


#ejercicio 4:
"""
run_ej4()
"""
#ejercicio 5.a:
""""
run_ej5_a(img=img, show=True, same_size=True)
"""
#ejercicio 5.b:
"""
a = run_ej5_b(img=img, show=True, same_size=False)
#a = a[:-2,:]
"""
#ejericicio 5.c:
"""
run_ej5_c(img=img, show=True)
"""
#ejercicio 5.d:

#A bilineal
"""
a = run_ej5_a(img=img, show=False)
b = run_ej5_a(img=img, show=False)
new_mat = np.hstack((a, b))
b = run_ej5_d(a, show=True, method = 'bilineal', scale = 4)
 """

#B bilineal
"""
a = run_ej5_b(img=img, show=False)
b = run_ej5_b(img=img, show=False)
new_mat = np.hstack((a, b))
b = run_ej5_d(a, show=True, method = 'bilineal', scale = 4)
"""
#C bilineal
"""
a = run_ej5_c(img=img, show=False)
b = run_ej5_c(img=img, show=False)
new_mat = np.hstack((a, b))
b = run_ej5_d(a, show=True, method = 'bilineal', scale = 4)
 """
#A bicubica
"""
a = run_ej5_a(img=img, show=False)
b = run_ej5_a(img=img, show=False)
new_mat = np.hstack((a, b))
b = run_ej5_d(a, show=True, method = 'bicubic', scale = 4)
"""

#B bicubica
"""
a = run_ej5_b(img=img, show=False)
b = run_ej5_b(img=img, show=False)
new_mat = np.hstack((a, b))
b = run_ej5_d(a, show=True, method = 'bicubic', scale = 4)
"""
#C bicubica
#"""
a = run_ej5_c(img=img, show=False)
b = run_ej5_c(img=img, show=False)
new_mat = np.hstack((a, b))
b = run_ej5_d(a, show=True, method = 'bicubic', scale = 4)
#"""


# c = cv.resize(src=a, dsize=(256, 256), fx=0, fy=0, interpolation=cv.INTER_CUBIC)
# cv.imshow(winname='Averaged', mat=np.hstack((c, b)))

cv.waitKey(0)
