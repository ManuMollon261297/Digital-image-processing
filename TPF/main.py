import cv2 as cv 
import numpy as np
import pytesseract 
import math
from scipy import ndimage

def orientation_correction(img):
    # Canny Algorithm for edge detection
    img_edges = cv.Canny(img, 100, 100, apertureSize=3)
    # Using Houghlines to detect lines
    lines = cv.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    
    # Finding angle of lines in polar coordinates
    angles = []
    print(lines)
    if lines is None:
        return img
    else:
        for x1, y1, x2, y2 in lines[0]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
    
        # Getting the median angle
        median_angle = np.median(angles)
    
        # Rotating the image with this median angle
        img_rotated = ndimage.rotate(img, median_angle)
        return img_rotated
        


def detectText(img):

    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray) 
    cv.waitKey(0)

    # Histogram Equalization
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(gray)
    cv.imshow('equalized', equ) 
    cv.waitKey(0)

    # Gaussian Blur
    blur = cv.GaussianBlur(equ, (9,9), cv.BORDER_DEFAULT)
    cv.imshow('blur', blur) 
    cv.waitKey(0)

    # Sobel
    grad_x = cv.Sobel(blur, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(blur, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    sobel = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    cv.imshow('sobel', sobel) 
    cv.waitKey(0)

    # Threshold
    _, thresh = cv.threshold(sobel, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('thresh', thresh) 
    cv.waitKey(0)

    kernel1 = cv.getStructuringElement(cv.MORPH_RECT, (6, 6))
    kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (20, 20))

    # opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel1)
    # cv.imshow('opening', opening) 
    # cv.waitKey(0)

    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel2)
    cv.imshow('closing', closing) 
    cv.waitKey(0)

    # Finding contours 
    contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 

    return contours, equ.copy()
  
# Mention the installed location of Tesseract-OCR in your system 
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
  
# Load image
img = cv.imread('sample3.png') 

cv.imshow('original', img) 
cv.waitKey(0)

contours, im2 = detectText(img)
  
# Creating File
file = open('recognized.txt', 'w+') 
file.write('') 
file.close()

cont = input()

if cont == 'exit':
    pass
else:
    for contour in contours: 
        x, y, w, h = cv.boundingRect(contour) 
          
        # Drawing a rectangle on copied image 
        #rect = cv.rectangle(im2, (x, y), (x + w, y + h), (0, 255, 0), 2) 
          
        # Cropping the text block for giving input to OCR 
        cropped = im2[y:y + h, x:x + w] 

        rotated = orientation_correction(cropped)

        #cv.imshow('cropped and rotated', rotated) 
        #cv.waitKey(0)
          
        file = open('recognized.txt', 'a') 
          
        # Apply OCR on the cropped image 
        text = pytesseract.image_to_string(rotated) 
         
        print(text) 
        file.write(text) 
        file.write('\n') 
        file.close