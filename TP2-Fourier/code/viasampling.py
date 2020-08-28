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
f1 = f1[1:500, 1:500]
print(f1.shape)
# f2 = downsample(f1, m, 'FILTER_OFF')
# f2 = downsample(f1, m, 'FILTER_ON')

f3 = upsample(f1, m)
imshow(mat=f1, winname='F1')
# imshow(mat=f2, winname='F2')
imshow(mat=f1, winname='F1')
cv.waitKey(0)

