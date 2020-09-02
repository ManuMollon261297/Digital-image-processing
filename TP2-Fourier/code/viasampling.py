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

