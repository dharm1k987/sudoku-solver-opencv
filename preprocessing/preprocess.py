import cv2


def preprocess(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    greyscale = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoise = cv2.GaussianBlur(greyscale, (9, 9), 0)
    thresh = cv2.adaptiveThreshold(denoise, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    inverted = cv2.bitwise_not(thresh, 0)
    morph = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    dilated = cv2.dilate(morph, kernel, iterations=1)
    return dilated


    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = (9, 9)
    blur = cv2.GaussianBlur(img_gray, kernel, 0)

    # determine threshold of a pixel based on small area surrounding it
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # invert it so the lines and numbers are white and area is black
    invert = cv2.bitwise_not(thresh, thresh)

    return invert
