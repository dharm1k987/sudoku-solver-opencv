import cv2
import numpy as np
import tensorflow as tf

from helpers import process_helpers


def create_grid_mask(vertical, horizontal):
    # combine the vertical and horizontal lines to make a grid
    grid = cv2.add(horizontal, vertical)
    # threshold and dilate the grid to cover more area
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)

    # find the list of where the lines are, this is an array of (rho, theta in radians)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    lines = process_helpers.draw_lines(grid, pts)
    # extract the lines so only the numbers remain
    mask = cv2.bitwise_not(lines)
    return mask


def get_grid_lines(img, length=10):
    horizontal = process_helpers.grid_line_helper(img, 1, length)
    vertical = process_helpers.grid_line_helper(img, 0, length)
    return vertical, horizontal


def find_contours(img, original):
    # find contours on thresholded image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort by the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None

    # make sure this is the one we are looking for
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
        num_corners = len(approx)

        if num_corners == 4 and area > 1000:
            polygon = cnt
            break

    if polygon is not None:
        # find its extreme corners
        top_left = process_helpers.find_extreme_corners(polygon, min, np.add)  # has smallest (x + y) value
        top_right = process_helpers.find_extreme_corners(polygon, max, np.subtract)  # has largest (x - y) value
        bot_left = process_helpers.find_extreme_corners(polygon, min, np.subtract)  # has smallest (x - y) value
        bot_right = process_helpers.find_extreme_corners(polygon, max, np.add)  # has largest (x + y) value

        # if its not a square, we don't want it
        if bot_right[1] - top_right[1] == 0:
            return []
        if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return []

        cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

        # draw corresponding circles
        [process_helpers.draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

        return [top_left, top_right, bot_right, bot_left]

    return []


def warp_image(corners, original):
    # we will be warping these points
    corners = np.array(corners, dtype='float32')
    top_left, top_right, bot_right, bot_left = corners

    # find the best side width, since we will be warping into a square, height = length
    width = int(max([
        np.linalg.norm(top_right - bot_right),
        np.linalg.norm(top_left - bot_left),
        np.linalg.norm(bot_right - bot_left),
        np.linalg.norm(top_left - top_right)
    ]))

    # create an array with shows top_left, top_right, bot_left, bot_right
    mapping = np.array([[0, 0], [width - 1, 0], [width - 1, width - 1], [0, width - 1]], dtype='float32')

    matrix = cv2.getPerspectiveTransform(corners, mapping)

    return cv2.warpPerspective(original, matrix, (width, width)), matrix


def split_into_squares(warped_img):
    squares = []

    width = warped_img.shape[0] // 9

    # find each square assuming they are of the same side
    for j in range(9):
        for i in range(9):
            p1 = (i * width, j * width)  # Top left corner of a bounding box
            p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box
            squares.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])

    return squares


def clean_squares(squares):
    cleaned_squares = []
    i = 0

    for square in squares:
        new_img, is_number = process_helpers.clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
            i += 1

        else:
            cleaned_squares.append(0)

    return cleaned_squares


def recognize_digits(squares_processed, model):
    s = ""
    formatted_squares = []
    location_of_zeroes = set()

    # img =
    blank_image = np.zeros_like(cv2.resize(squares_processed[0], (32, 32)))

    for i in range(len(squares_processed)):
        if type(squares_processed[i]) == int:
            location_of_zeroes.add(i)
            formatted_squares.append(blank_image)
        else:
            img = cv2.resize(squares_processed[i], (32, 32))
            formatted_squares.append(img)

    formatted_squares = np.array(formatted_squares)
    all_preds = list(map(np.argmax, model(tf.convert_to_tensor(formatted_squares))))
    for i in range(len(all_preds)):
        if i in location_of_zeroes:
            s += "0"
        else:
            s += str(all_preds[i] + 1)

    return s


def draw_digits_on_warped(warped_img, solved_puzzle, squares_processed):
    width = warped_img.shape[0] // 9

    img_w_text = np.zeros_like(warped_img)

    # find each square assuming they are of the same side
    index = 0
    for j in range(9):
        for i in range(9):
            if type(squares_processed[index]) == int:
                p1 = (i * width, j * width)  # Top left corner of a bounding box
                p2 = ((i + 1) * width, (j + 1) * width)  # Bottom right corner of bounding box

                center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                text_size, _ = cv2.getTextSize(str(solved_puzzle[index]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 4)
                text_origin = (center[0] - text_size[0] // 2, center[1] + text_size[1] // 2)

                cv2.putText(warped_img, str(solved_puzzle[index]),
                            text_origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            index += 1

    return img_w_text


def unwarp_image(img_src, img_dest, pts, time):
    pts = np.array(pts)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
                          dtype='float32')
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    cv2.fillConvexPoly(img_dest, pts, 0, 16)

    dst_img = cv2.add(img_dest, warped)

    dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]
    cv2.putText(dst_img, time, (dst_img_width - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

    return dst_img
