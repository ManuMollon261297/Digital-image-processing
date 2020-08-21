import numpy as np
import cv2 as cv

def biDimInterpol(imageArray, reScaleFactor = 4):
    imageArray = imageArray.astype(np.float64)
    imageArray = np.pad(imageArray, 1, 'edge')
    width = imageArray.shape[0]
    height = imageArray.shape[1]
    nX = width-2
    nY = height-2
    newImageArray = np.zeros((nX*reScaleFactor, nY*reScaleFactor))
    nXp = nX*reScaleFactor
    nYp = nY*reScaleFactor
    eX = nX-1
    eY = nY-1
    iX = (nXp-nX)//eX
    iY = (nYp-nY)//eY
    xX = (nXp-nX)%eX
    xY = (nYp-nY)%eY
    extraX=1
    extraY=1
    for i in range(width-3):
      for j in range(height-3):
        smallArray = imageArray[i:i+4,j:j+4]
        arrayF = getArrayF(smallArray)
        coefs = getCoefs(arrayF)
        if(xX == 0):
          extraX = 0
        if(xY == 0):
          extraY = 0
        for k in range(2+iX+extraY):
          for l in range(2+iY+extraX):
            x = k/(1+iX+extraX)
            y = l/(1+iY+extraX)
            if interpol(coefs,x,y) > 255:
              interpolVal = 255
            else:
              interpolVal = interpol(coefs,x,y)
            newImageArray[i*reScaleFactor+k+(nYp-nY)%eY-xY][j*reScaleFactor+l+(nXp-nX)%eX-xX] = interpolVal
        if xX>0:
          xX = xX-1
      if xY>0:
        xY = xY-1
      extraX=1
      xX = (nXp-nX)%eX
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