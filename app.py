import cv2

from current import process, sudoku
from helpers import display
from models import model_wrapper
from preprocessing import preprocess
import copy
import time
import numpy as np

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

my_model = model_wrapper.model_wrapper(None, False, "weights.h5", "model.hdf5")

prev = 0

already_solved = []
unsolved = []

while True:
    # time_elapsed = time.time() - prev

    success, img = cap.read()

    if True:
        # prev = time.time()

        img_result = img.copy()
        img_result2 = img.copy()
        img_corners = img.copy()

        processed_img = preprocess.preprocess(img.copy())
        corners = process.find_contours(processed_img, img_corners)

        if corners:
            warped, matrix = process.warp_image(corners, img)
            # img_result = warped
            warped_processed = preprocess.preprocess(warped)

            vertical_lines, horizontal_lines = process.get_grid_lines(warped_processed)
            mask = process.create_grid_mask(vertical_lines, horizontal_lines)
            numbers = cv2.bitwise_and(warped_processed, mask)
            img_result2 = numbers



            squares = process.split_into_squares(numbers)
            squares_processed = process.clean_squares(squares)

            # squares_processed_before = copy.deepcopy(squares_processed)

            squares_num_array, s = process.recognize_digits(squares_processed, my_model)

            if not any(np.array_equal(s, x) for x in unsolved):
                solved_puzzle, time = sudoku.solve(s)
                if solved_puzzle is not None:
                    # already_solved.append(squares_num_array)
                    process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                    img_result = process.unwarp_image(warped, img_result, corners, time)
                else:
                    unsolved.append(s)

            else:
                print('Unsolvable, moving on')

        cv2.imshow('window', display.stackImages(0.75, [img_corners, img_result2, img_result]))

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
