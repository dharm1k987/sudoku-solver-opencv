import operator
import cv2


def find_extreme_corners(polygon, limit_fn, compare_fn):
    # limit_fn is the min or max function
    # compare_fn is the np.add or np.subtract function

    # if we are trying to find bottom left corner, we know that it will have the smallest (x - y) value
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return (polygon[section][0][0], polygon[section][0][1])


def draw_extreme_corners(pts, original):
    cv2.circle(original, pts, 10, (0, 255, 0), 10)
