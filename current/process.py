import cv2
import numpy as np
import operator
from helpers import process_helpers
from current import sudoku

def create_grid_mask(vertical, horizontal):
    grid = cv2.add(horizontal, vertical)
    grid = cv2.adaptiveThreshold(grid, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 235, 2)
    grid = cv2.dilate(grid, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=2)
    pts = cv2.HoughLines(grid, .3, np.pi / 90, 200)

    def draw_lines(im, pts):
        im = np.copy(im)
        pts = np.squeeze(pts)
        for r, theta in pts:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * r
            y0 = b * r
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * a)
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * a)
            cv2.line(im, (x1, y1), (x2, y2), (255, 255, 255), 4)
        return im

    lines = draw_lines(grid, pts)
    mask = cv2.bitwise_not(lines)
    return mask

def get_grid_lines(img, length=10):
    horizontal = np.copy(img)
    cols = horizontal.shape[1]
    horizontal_size = cols // length
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    vertical = np.copy(img)
    rows = vertical.shape[0]
    vertical_size = rows // length
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

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
            # square = warped_img[p1[1]:p2[1], p1[0]:p2[0]]
            # squares[j][i] = square
            squares.append(warped_img[p1[1]:p2[1], p1[0]:p2[0]])

    return squares


def clean_squares(squares):
    cleaned_squares = []
    i = 0

    for square in squares:
        new_img, is_number = process_helpers.clean_helper(square)

        if is_number:
            cleaned_squares.append(new_img)
            # cv2.imwrite('{}.png'.format(i), new_img)
            i += 1

        else:
            cleaned_squares.append(0)

    return cleaned_squares
    # for j in range(9):
    #     for i in range(9):
    #         # clean up the img at squares[j][i]
    #         new_img, is_number = process_helpers.clean_helper(squares[j][i])
    #         if is_number:
    #             squares[j][i] = new_img
    #             cv2.imwrite('{}-{}.png'.format(j,i), squares[j][i])
    #         else:
    #             squares[j][i] = -1
    #
    # return squares


def recognize_digits(squares_processed, model):
    squares_digits = []
    s = ""
    i = 0
    for square in squares_processed:
        if type(square) == int:
            s += "0"
            squares_digits.append(0)
        else:

            i += 1

            img = square
            img = img.reshape(img.shape[0], img.shape[0])
            img = cv2.resize(img, (32, 32))
            # cv2.imwrite('{}.png'.format(i), img)

            img = img.reshape(img.shape[0], img.shape[0], 1)
            img = np.expand_dims(img, axis=0)
            pred = np.argmax(model.predict(np.vstack([img]))) + 1
            squares_digits.append(pred)
            s += str(pred)


            # resized = cv2.resize(square, (32, 32), interpolation=cv2.INTER_LANCZOS4)
            # array = np.array([resized])
            # reshaped = array.reshape(array.shape[0], 32, 32, 1)
            # cv2.imwrite('{}.png'.format(i), resized)
            #
            # flt = reshaped.astype('float32')
            # flt /= 255
            # prediction = model.predict_classes(flt)
            # squares_digits.append(prediction[0] + 1)  # OCR predicts from 0-8, changing it to 1-9

    # for j in range(9):
    #     for i in range(9):
    #         if type(squares_processed[j][i]) == int:
    #             pass
    #         else:
    #             img = squares_processed[j][i]
    #             img = img.reshape(img.shape[0], img.shape[0])
    #             img = cv2.resize(img, (32, 32))
    #             img = img.reshape(img.shape[0], img.shape[0], 1)
    #             img = np.expand_dims(img, axis=0)
    #             squares_processed[j][i] = np.argmax(model.predict(np.vstack([img]))) + 1

    # squares_digits = np.array(squares_digits).reshape((9, 9))

    # sudoku.print_grid(squares_digits)
    return squares_digits, s


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

    cv2.putText(dst_img, time, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    return dst_img
