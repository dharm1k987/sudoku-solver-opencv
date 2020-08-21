import cv2
import numpy as np
from preprocessing import preprocess
from helpers import display


img = cv2.imread("images/sudoku.jpg")

processed_img = preprocess.preprocess(img.copy())

cv2.imshow('window', display.stackImages(0.50, [img, processed_img]))

cv2.waitKey(0)
