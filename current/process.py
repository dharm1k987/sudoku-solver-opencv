import cv2
import numpy as np
import operator
from helpers import process_helpers


def find_contours(img, original):
    # find contours on thresholded image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort by the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None

    # make sure this is the one we are looking for
    for cnt in contours:
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
        num_corners = len(approx)

        if num_corners == 4:
            polygon = cnt
            break

    if polygon is not None:
        cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

        # find its extreme corners
        top_left = process_helpers.find_extreme_corners(polygon, min, np.add)  # has smallest (x + y) value
        top_right = process_helpers.find_extreme_corners(polygon, max, np.subtract)  # has largest (x - y) value
        bot_left = process_helpers.find_extreme_corners(polygon, min, np.subtract)  # has smallest (x - y) value
        bot_right = process_helpers.find_extreme_corners(polygon, max, np.add)  # has largest (x + y) value

        print(top_left)

        # draw corresponding circles
        [process_helpers.draw_extreme_corners(x, original) for x in [top_left, top_right, bot_left, bot_right]]

        return [top_left, top_right, bot_left, bot_right]

    return []


def warp_image(corners, original):
    # we will be warping these points
    corners = np.array(corners, dtype='float32')
    top_left, top_right, bot_left, bot_right = corners

    # find the best side width, since we will be warping into a square, height = length
    width = int(max([
        np.linalg.norm(top_right - bot_right),
        np.linalg.norm(top_left - bot_left),
        np.linalg.norm(bot_right - bot_left),
        np.linalg.norm(top_left - top_right)
    ]))

    # create an array with shows top_left, top_right, bot_left, bot_right
    mapping = np.array([[0, 0], [width - 1, 0], [0, width - 1], [width - 1, width - 1]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(corners, mapping)

    return cv2.warpPerspective(original, matrix, (width, width))

def split_into_squares(warped_img):
    squares = [[0]*9 for i in range(9)]

    width = warped_img.shape[0] // 9
    print(warped_img.shape)

    # find each square assuming they are of the same side
    for j in range(9):
        for i in range(9):
            p1 = (i * width, j * width)  # Top left corner of a bounding box
            p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box
            square = warped_img[p1[1]:p2[1], p1[0]:p2[0]]
            # print(warped_img[p1[1]:p2[1], p1[0]:p2[1]])
            squares[j][i] = square
            # cv2.imwrite('{}-{}.png'.format(j,i), squares[j][i])


    return squares

def clean_squares(squares):
    for j in range(9):
        for i in range(9):
            # clean up the img at squares[j][i]
            new_img, is_number = process_helpers.clean_helper(squares[j][i])
            if is_number:

                # image_center = tuple(np.array(new_img.shape[1::-1]) / 2)
                # rot_mat = cv2.getRotationMatrix2D(image_center, -5, 1.0)
                # new_img = cv2.warpAffine(new_img, rot_mat, new_img.shape[1::-1], flags=cv2.INTER_LINEAR)

                squares[j][i] = new_img
                cv2.imwrite('{}-{}.png'.format(j,i), squares[j][i])
            else:
                squares[j][i] = -1

    return squares


def recognize_digits(squares_processed, model):
    for j in range(9):
        for i in range(9):
            if type(squares_processed[j][i]) == int:
                pass
            else:
                img = squares_processed[j][i]
                img = img.reshape(img.shape[0], img.shape[0])
                img = cv2.resize(img, (28, 28))
                squares_processed[j][i] = np.argmax(model.predict(img.reshape(1, 28, 28)))

                # img = img.reshape(img.shape[0], img.shape[0])
                # img = cv2.resize(img, (28, 28))
                # cv2.imshow('w', img)
                # print(np.argmax(    my_model.predict(img.reshape(1, 28, 28))    ) + 1)
                #

    for j in range(9):
        for i in range(9):
            print(squares_processed[j][i], end ="\t")
        print()
