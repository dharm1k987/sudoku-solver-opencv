import cv2
import os
import numpy as np




for file in os.listdir("./original"):

    img = cv2.imread('{}/{}'.format('original', file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, (28, 28))
    remove_dot = file.split('.')[0]
    img_num = int(remove_dot.split('-')[0])
    img_num_second = int(remove_dot.split('-')[1])


    # perform the following modifications

    # rotate left
    image_center = tuple(np.array(img.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, 7, 1.0)
    rotate_left = cv2.warpAffine(img.copy(), rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    # rotate right
    rot_mat = cv2.getRotationMatrix2D(image_center, -7, 1.0)
    rotate_right = cv2.warpAffine(img.copy(), rot_mat, img.shape[1::-1], flags=cv2.INTER_LINEAR)

    # dilate
    kernel = np.array([[0, 0.8, 0], [0.8, 0.8, 0.8], [0, 0.8, 0]], np.uint8)
    dilate = cv2.dilate(img.copy(), kernel)

    # erode
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)
    erode = cv2.erode(img.copy(), kernel)

    # zoom
    zoom = cv2.resize(img.copy(), None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    zoom_out = cv2.resize(img.copy(), None, fx=0.75, fy=0.75, interpolation=cv2.INTER_AREA)

    # blur
    blur = cv2.GaussianBlur(img.copy(), (5,5), 0)


    to_write = [
        rotate_left,
        rotate_right,
        dilate,
        erode,
        zoom,
        zoom_out,
        blur
    ]

    starting = 218

    for f in to_write:
        cv2.imwrite('new/{}-{}.png'.format(img_num, img_num_second + starting), f)
        starting += 1




