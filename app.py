import cv2

from current import process, sudoku
from helpers import display
from models import model_wrapper
from preprocessing import preprocess
import copy
import time

frameWidth = 960
frameHeight = 720

# change to 1 if using USB webcam
cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

my_model = model_wrapper.model_wrapper(None, False, "weights.h5", "model-saved")

prev = 0

while True:
    # time_elapsed = time.time() - prev

    success, img = cap.read()

    if True:
        # prev = time.time()

        img_result = img.copy()

        processed_img = preprocess.preprocess(img.copy())
        corners = process.find_contours(processed_img, img)

        if corners:
            warped, matrix = process.warp_image(corners, img)
            warped_processed = preprocess.preprocess(warped)
            squares = process.split_into_squares(warped_processed)
            squares_processed = process.clean_squares(squares)
            squares_processed_before = copy.deepcopy(squares_processed)

            squares_num_array = process.recognize_digits(squares_processed, my_model)
            solved_puzzle, time = sudoku.solve(squares_num_array)
            if solved_puzzle:
                process.draw_digits_on_warped(warped, solved_puzzle, squares_processed_before)
                img_result = process.unwarp_image(warped, img_result, corners, time)

        cv2.imshow('window', display.stackImages(0.5, [img, img_result, processed_img]))

    wait = cv2.waitKey(1)
    if wait & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()











#
# img = cv2.imread("imgs/ex2.jpg")
# img_result = img.copy()
#
# processed_img = preprocess.preprocess(img.copy())
# corners = process.find_contours(processed_img, img.copy())
#
# if corners:
#     warped, matrix = process.warp_image(corners, img)
#     warped_processed = preprocess.preprocess(warped)
#     squares = process.split_into_squares(warped_processed)
#     squares_processed = process.clean_squares(squares)
#     squares_processed_before = copy.deepcopy(squares_processed)
#
#     squares_num_array = process.recognize_digits(squares_processed, my_model)
#     solved_puzzle, time = sudoku.solve(squares_num_array)
#     if solved_puzzle:
#         process.draw_digits_on_warped(warped, solved_puzzle, squares_processed_before)
#         img_result = process.unwarp_image(warped, img_result, corners, time)
#
#     cv2.imshow('window', display.stackImages(0.50, [img_result]))
#
# cv2.waitKey(0)
