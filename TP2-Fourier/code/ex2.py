import cv2 as cv
from scipy.ndimage import correlate
import numpy as np

h = np.array([[0, 1/6, 0], [1/6, 1/3, 1/6], [0, 1/6, 0]])
h = h/np.sum(h)
# h = fspecial('unsharp')
# h = fspecial('disk')

img = cv.imread(filename = '../data/images/barbara.png', flags = cv.IMREAD_GRAYSCALE)
print(img.shape)

h1 = correlate(img, h) # make conv with the filter
new_mat = np.hstack((img, h1))

cv.imshow(mat = new_mat, winname = 'Original Image')
cv.waitKey(0)