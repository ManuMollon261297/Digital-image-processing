import cv2 as cv
from upsampling import upsample
# from downsampling import downsample
from cv2 import imread, imshow
import numpy as np

# Via sampling theory
m = 2

cap = cv.VideoCapture('barbara.gif')
ret, f1 = cap.read()
f1 = cv.cvtColor(src=f1, code=cv.COLOR_BGR2GRAY)
f1 = f1[250:500, 250:500]

# f2 = downsample(f1, m, 'FILTER_OFF')
# f2 = downsample(f1, m, 'FILTER_ON')

f3 = upsample(f1, m)
imshow(mat=f1, winname='F1')
imshow(mat=f3.astype(np.uint8), winname='F3')
cv.waitKey(0)

# e1 = 0.8
# e2 = 0.8
# x = np.array([0, 0, 0, 0.5, 0, 0, 0, 0.5, 0, 0, 0, 0.5], dtype=np.float)
# y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float)
# for n in range(3, len(x)):
#     y[n] = x[n] + e1 * (x[n-2] + e2 * y[n-2])
# print(y)
