import cv2 as cv 
import numpy as np
import pytesseract 
import math
import pyttsx3
from scipy import ndimage
from googletrans import Translator

translator = Translator()
engine = pyttsx3.init()

# Mention the installed location of Tesseract-OCR in your system 
pytesseract.pytesseract.tesseract_cmd = 'C:\\Users\\Manuel Mollon\\Dropbox\\Q2-IMAG\\Digital-image-processing\\TPF\\Tesseract-OCR\\tesseract.exe'

def orientationCorrection(img):
    # Canny Algorithm
    img_edges = cv.Canny(img, 100, 100, apertureSize=3)
    #cv.imshow('original', img_edges) 
    #cv.waitKey(0)

    # HoughLines Transform
    lines = cv.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)
    
    # Find angle of lines
    angles = []
    if lines is None:
        return img
    else:
        for x1, y1, x2, y2 in lines[0]:
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
            cv.line(img_edges, (x1, y1), (x2, y2), 255, 2)
    
        #cv.imshow('Detected Lines', img_edges) 
        #cv.waitKey(0)

        # Median angle
        median_angle = np.median(angles)
    
        # Rotating image
        img_rotated = ndimage.rotate(img, median_angle)
        return img_rotated
        


def detectText(img):

    # Denoise
    #den =cv.fastNlMeansDenoisingColored(img,None,100,100,7,21)
    #cv.imshow('denoised', cv.resize(den,(700,700))) 
    #cv.waitKey(0)

    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', cv.resize(gray,(700,700))) 
    cv.waitKey(0)

    # Histogram Equalization
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    equ = clahe.apply(gray)
    cv.imshow('equalized', cv.resize(equ,(700,700))) 
    cv.waitKey(0)

    # Gaussian Blur
    sigmaX = 10 #10
    sigmaY = 10 #10
    ker_size = 201
    blur = cv.GaussianBlur(equ, (ker_size, ker_size), sigmaX, sigmaY, cv.BORDER_DEFAULT)
    cv.imshow('blur', cv.resize(blur,(700,700))) 
    cv.waitKey(0)

    # Sobel
    grad_x = cv.Sobel(blur, cv.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(blur, cv.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv.BORDER_DEFAULT)
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    sobel = cv.addWeighted(abs_grad_x, 0, abs_grad_y, 1, 0)
    cv.imshow('sobel', cv.resize(sobel,(700,700))) 
    cv.waitKey(0)

    # Threshold
    _, thresh = cv.threshold(sobel, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('thresh', cv.resize(thresh,(700,700))) 
    cv.waitKey(0)

    x, y = thresh.shape 
    length = int(math.sqrt(x*y)*0.022)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (length, length))

    closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
    cv.imshow('closing', cv.resize(closing,(700,700))) 
    cv.waitKey(0)

    # Finding contours 
    contours, _ = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE) 

    return contours, gray.copy()

def imageToVoiceAndText(image_add, voice=True):
    global translator
    global engine

    # Load image
    img = cv.imread(image_add) 

    cv.imshow('original', cv.resize(img,(700,700))) 
    cv.waitKey(0)

    contours, im2 = detectText(img)
      
    # Creating File
    file = open('recognized.txt', 'w+') 
    file.write('') 
    file.close()

    print('Continue? (y/n)')
    cont = 'y'#input()

    if cont == 'y':
        for contour in reversed(contours): 
            x, y, w, h = cv.boundingRect(contour) 
              
            # Cropping the text block for giving input to OCR 
            cropped = im2[y:y + h, x:x + w]

            rotated = orientationCorrection(cropped) 
              
            # Apply OCR on the cropped image 
            text = pytesseract.image_to_string(rotated) 

            length = len(text) - text.count(' ')

            if length > 3 and not text.isspace():
                # Drawing a rectangle on copied image 
                rect = cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 15) 
                #cv.imshow('cropped and rotated', cv.resize(rotated,(700,700))) 
                #cv.waitKey(0)
                file = open('recognized.txt', 'a') 
                print('New Contour:')
                print('Text:', text)
                print('Length:', len(text))
                det = translator.detect(text)
                print('Language:', det.lang)
                print('Confidence:', det.confidence)
                translation = translator.translate(text, dest='en').text
                print('Translation:', translation)
                if voice:
                    engine.say(translation)
                    engine.runAndWait()
                file.write(translation) 
                file.write('\n') 
                file.close

    cv.imshow('Detected Texts', cv.resize(img,(700,700))) 
    cv.waitKey(0)

ext = 'images/'
samples =   ['1_sCh2dv4mhn-xqjWB3d4Tuw.gif', 'IMG_8766.jpg', 'IMG_8793.jpg', 'IMG_8794.jpg',
             'IMG_8795.jpg', 'IMG_8796.jpg', 'sample6.png', 'sample7.jpg',
             'sample8.jpg', 'Screenshot 2020-12-19 16.50.09.png', 'uOn7S.jpg','_96403811_hi012217444.jpg']

for sample in samples:
    imageToVoiceAndText(ext + sample, voice=False)