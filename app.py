import time as t

import cv2

from current import process, sudoku
from models import model_wrapper
from preprocessing import preprocess

frameWidth = 960
frameHeight = 720

cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

# load the model with weights
my_model = model_wrapper.model_wrapper(None, False, None, "model.hdf5")

prev = 0

seen = dict()

while True:
    time_elapsed = t.time() - prev

    success, img = cap.read()

    if time_elapsed > 1. / frame_rate:
        prev = t.time()

        img_result = img.copy()
        img_corners = img.copy()

        processed_img = preprocess.preprocess(img)
        corners = process.find_contours(processed_img, img_corners)

        if corners:
            warped, matrix = process.warp_image(corners, img)
            warped_processed = preprocess.preprocess(warped)

            vertical_lines, horizontal_lines = process.get_grid_lines(warped_processed)
            mask = process.create_grid_mask(vertical_lines, horizontal_lines)
            numbers = cv2.bitwise_and(warped_processed, mask)

            squares = process.split_into_squares(numbers)
            squares_processed = process.clean_squares(squares)

            squares_guesses = process.recognize_digits(squares_processed, my_model)

            # if it is impossible, continue
            if squares_guesses in seen and seen[squares_guesses] is False:
                continue

            # if we already solved this puzzle, just fetch the solution
            if squares_guesses in seen:
                process.draw_digits_on_warped(warped, seen[squares_guesses][0], squares_processed)
                img_result = process.unwarp_image(warped, img_result, corners, seen[squares_guesses][1])

            else:
                solved_puzzle, time = sudoku.solve_wrapper(squares_guesses)
                if solved_puzzle is not None:
                    process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                    img_result = process.unwarp_image(warped, img_result, corners, time)
                    seen[squares_guesses] = [solved_puzzle, time]

                else:
                    seen[squares_guesses] = False

    cv2.imshow('window', img_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
