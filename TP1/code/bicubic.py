import numpy as np
import cv2 as cv

def biDimInterpol(imageArray, reScaleFactor = 4):
    #cv.imshow(winname='Original Imagen', mat=imageArray.astype(np.uint8))
    #cv.waitKey(0)
    imageArray = imageArray.astype(np.float64)
    width = imageArray.shape[0]
    height = imageArray.shape[1]
    imageArray = np.pad(imageArray, 1, 'edge')
    increment = 1/(2*reScaleFactor-1)
    newImageArray = np.zeros((width*reScaleFactor,height*reScaleFactor))
    for i in range(int(width/2)):
      for j in range(int(height/2)):
        smallArray = imageArray[i*2:i*2+4,j*2:j*2+4]  
        arrayF = getArrayF(smallArray)
        coefs = getCoefs(arrayF)
        intervals = 2*reScaleFactor
        for k in range(intervals):
          for l in range(intervals):   
            x = k*increment
            y = l*increment
            interpolVal = interpol(coefs,x,y)
            newImageArray[i*2*reScaleFactor+k][j*2*reScaleFactor+l] = interpolVal
    return newImageArray.astype(np.uint8)

def getArrayF(smallArray):
    arrayF = np.array([[smallArray[1][1],smallArray[1][2],fy(smallArray,1,1),fy(smallArray,1,2)],
                       [smallArray[2][1],smallArray[2][2],fy(smallArray,2,1),fy(smallArray,2,2)],
                       [fx(smallArray,1,1),fx(smallArray,1,2),fxy(smallArray,1,1),fxy(smallArray,1,2)],
                       [fx(smallArray,2,1),fx(smallArray,2,2),fxy(smallArray,2,1),fxy(smallArray,2,2)]])
    return arrayF

def getCoefs(matrix):
    return np.dot(np.dot(getCoefs.array1, matrix), getCoefs.array2)
getCoefs.array1 = np.array([[1,0,0,0],
                            [0,0,1,0],
                            [-3,3,-2,-1],
                            [2,-2,1,1]])
getCoefs.array2 = np.array([[1,0,-3,2],
                            [0,0,3,-2],
                            [0,1,-2,1],
                            [0,0,-1,1]])

def fx(smallArray,x,y):
    return (smallArray[x+1][y]-smallArray[x-1][y])/2

def fy(smallArray,x,y):
    return (smallArray[x][y+1]-smallArray[x][y-1])/2

def fxy(smallArray,x,y):
    return (smallArray[x+1][y+1]-smallArray[x+1][y-1]-smallArray[x-1][y+1]+smallArray[x-1][y-1])/4

def interpol(coefsArray,x,y):
    xArray = np.array([1,x,x**2,x**3])
    yArray = np.array([1,y,y**2,y**3])
    return np.dot(np.dot(xArray, coefsArray), yArray)

# img = cv.imread(filename='../mono.bmp', flags=cv.IMREAD_GRAYSCALE)
# cv.imshow(winname='Original Imagen', mat=img.astype(np.uint8))
# cv.waitKey(0)
# N = 15
# auxList = list(range(225))
# #array = np.array(auxList)
# #array = np.reshape(array,(N,N))
# #array = np.array([[1,2,3,4],
# #                  [2,4,6,8],
# #                  [3,6,9,12],
# #                  [4,8,12,16]])
# print(img)
# interpolArray = bicubic(img, reScaleFactor = 2)
# print(interpolArray.astype(np.uint8))
# cv.imshow(winname='Original Imagen', mat=img.astype(np.uint8))
# cv.waitKey(0)
# cv.imshow(winname='Bicubic Interpolation Image', mat=interpolArray.astype(np.uint8))
# cv.waitKey(0)

    
