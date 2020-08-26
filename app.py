import time as t

import cv2

from current import process, sudoku
from helpers import display
from models import model_wrapper
from preprocessing import preprocess
from collections import deque
from multiprocessing.pool import ThreadPool

frameWidth = 960
frameHeight = 720

cap = cv2.VideoCapture(0)
frame_rate = 30

# width is id number 3, height is id 4
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# change brightness to 150
cap.set(10, 150)

my_model = model_wrapper.model_wrapper(None, False, "weights.h5", "model.hdf5")

prev = 0

seen = dict()





# def process_frame(img):
#     # print(len(seen))
#     # some intensive computation...
#     img_result = img.copy()
#     img_corners = img.copy()
#
#     processed_img = preprocess.preprocess(img)
#     corners = process.find_contours(processed_img, img_corners)
#
#     if corners:
#         warped, matrix = process.warp_image(corners, img)
#         warped_processed = preprocess.preprocess(warped)
#
#         vertical_lines, horizontal_lines = process.get_grid_lines(warped_processed)
#         mask = process.create_grid_mask(vertical_lines, horizontal_lines)
#         numbers = cv2.bitwise_and(warped_processed, mask)
#
#         squares = process.split_into_squares(numbers)
#         squares_processed = process.clean_squares(squares)
#
#         squares_guesses = process.recognize_digits(squares_processed, my_model)
#
#         # if it is impossible, continue
#         if squares_guesses in seen and seen[squares_guesses] is False:
#             return img
#
#         # if we already solved this puzzle, just fetch the solution
#         if squares_guesses in seen:
#             process.draw_digits_on_warped(warped, seen[squares_guesses][0], squares_processed)
#             img_result = process.unwarp_image(warped, img_result, corners, seen[squares_guesses][1])
#             return img_result
#
#         else:
#             solved_puzzle, time = sudoku.solve(squares_guesses)
#             # print('Solving puzzle Good : {} took {}s'.format(solved_puzzle is not None, t.time() - start))
#             if solved_puzzle is not None:
#                 process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
#                 img_result = process.unwarp_image(warped, img_result, corners, time)
#                 seen[squares_guesses] = [solved_puzzle, time]
#
#             else:
#                 seen[squares_guesses] = False
#
#         return img_result
#
#     return img_corners
#
#
# thread_num = cv2.getNumberOfCPUs()
# pool = ThreadPool(processes=thread_num)
# pending_task = deque()
#
# while True:
#     while len(pending_task) > 0 and pending_task[0].ready():
#         res = pending_task.popleft().get()
#         cv2.imshow('Result', res)
#
#     if len(pending_task) < thread_num:
#         frame_got, frame = cap.read()
#         if frame_got:
#             task = pool.apply_async(process_frame, (frame.copy(),))
#             pending_task.append(task)
#
#     if cv2.waitKey(1) == ord('q') or not frame_got:
#         break




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
                solved_puzzle, time = sudoku.solve(squares_guesses)
                if solved_puzzle is not None:
                    process.draw_digits_on_warped(warped, solved_puzzle, squares_processed)
                    img_result = process.unwarp_image(warped, img_result, corners, time)
                    seen[squares_guesses] = [solved_puzzle, time]

                else:
                    seen[squares_guesses] = False

        # cv2.imshow('window', display.stackImages(0.90, [img_corners, img_result]))
        cv2.imshow('window', img_result)

    wait = cv2.waitKey(1)
    if wait & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

