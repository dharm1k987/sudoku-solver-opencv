import cv2
import numpy as np
from preprocessing import preprocess
from current import process
from helpers import display


img = cv2.imread("imgs/sudoku.jpg")

processed_img = preprocess.preprocess(img.copy())

process.find_contours(processed_img, img)

cv2.imshow('window', display.stackImages(0.50, [img, processed_img]))

cv2.waitKey(0)
