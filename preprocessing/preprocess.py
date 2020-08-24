import cv2


def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = (9, 9)
    blur = cv2.GaussianBlur(img_gray, kernel, 0)

    # determine threshold of a pixel based on small area surrounding it
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # invert it so the lines and numbers are white and area is black
    invert = cv2.bitwise_not(thresh, thresh)

    return invert
