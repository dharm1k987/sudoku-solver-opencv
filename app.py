import cv2
import numpy as np
from preprocessing import preprocess
from current import process
from helpers import display


img = cv2.imread("imgs/sudoku.jpg")

processed_img = preprocess.preprocess(img.copy())

corners = process.find_contours(processed_img, img.copy())

if corners:
    warped = process.warp_image(corners, img)
    cv2.imshow('window', display.stackImages(0.50, [img, processed_img, warped]))

cv2.waitKey(0)
