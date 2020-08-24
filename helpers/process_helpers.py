import operator
import cv2
import numpy as np


def find_extreme_corners(polygon, limit_fn, compare_fn):
    # limit_fn is the min or max function
    # compare_fn is the np.add or np.subtract function

    # if we are trying to find bottom left corner, we know that it will have the smallest (x - y) value
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return polygon[section][0][0], polygon[section][0][1]


def draw_extreme_corners(pts, original):
    cv2.circle(original, pts, 10, (0, 255, 0), 10)


def clean_helper(img):
    mid = img.shape[0] // 2
    ten_per = int(img.shape[0] * 0.1)
    twenty_per = int(img.shape[0] * 0.2)

    # automatically remove a border of 10% around the image if we have one
    img[0:ten_per, :] = 0
    img[img.shape[0] - ten_per:img.shape[0] - 1, :] = 0
    img[:, 0:ten_per] = 0
    img[:, img.shape[0] - ten_per:img.shape[0] - 1] = 0

    # find contours
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    main_contour = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # the main counter should roughly in the middle and vertical
        if h / w > 1 and not ((0.90 < h / w < 1.1) or (0.90 < w / h < 1.1)) and ten_per <= x and \
                img.shape[0] - ten_per >= x + ten_per:
            main_contour = cnt
            break

    if main_contour is not None:
        x, y, w, h = cv2.boundingRect(main_contour)

        # create new image where everything around bounding box is black
        new_img = np.zeros_like(img)
        for j in range(new_img.shape[0]):
            for i in range(new_img.shape[1]):
                if x <= i <= x + w and y <= j <= y + h:
                    new_img[j][i] = img[j][i]

        # if the center is mainly black, then we got here by accident so set to all black
        if np.isclose(img[mid - twenty_per:mid + twenty_per, mid - twenty_per:mid + twenty_per], 0).sum() \
                / img[mid - twenty_per:mid + twenty_per, mid - twenty_per:mid + twenty_per].size >= 0.90:
            return np.zeros_like(img), False

        else:
            return new_img, True

    return np.zeros_like(img), False
