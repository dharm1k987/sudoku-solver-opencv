import cv2

from current import process, sudoku
from models import model_wrapper
from preprocessing import preprocess

my_model = model_wrapper.model_wrapper(None, False, "weights.h5", "temp2-model")
# my_model.save('model_file')
img = cv2.imread("imgs/sudoku.jpg")

processed_img = preprocess.preprocess(img.copy())

corners = process.find_contours(processed_img, img.copy())

if corners:
    warped = process.warp_image(corners, img)
    warped_processed = preprocess.preprocess(warped)
    squares = process.split_into_squares(warped_processed)
    squares_processed = process.clean_squares(squares)
    squares_num_array = process.recognize_digits(squares_processed, my_model)
    solved_puzzle = sudoku.solve(squares_num_array)



# cv2.imshow('window', display.stackImages(0.50, [[img, processed_img],[warped, warped_processed]]))
#     cv2.imshow('window', squares[1][3])


cv2.waitKey(0)
