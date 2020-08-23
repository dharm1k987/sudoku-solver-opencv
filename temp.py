import cv2
import numpy as np
import os

# img = cv2.imread('0-7.png')
# imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# ten_per = int(img.shape[0] * 0.1)
#
# # automatically remove a border of 10% around the image if we have one
# img[0:ten_per, :] = 0
# img[img.shape[0] - ten_per:img.shape[0] - 1, :] = 0
# img[:, 0:ten_per] = 0
# img[:, img.shape[0] - ten_per:img.shape[0] - 1] = 0
#
#
#
# contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)
# x, y, w, h = cv2.boundingRect(contours[1])
#
# mid = img.shape[0]//2
#
# print(h/w, ten_per <= x and img.shape[0] - ten_per >= x + ten_per)
#
# cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
# cv2.imshow('w', img)
# cv2.waitKey(0)


for file in os.listdir('./'):
    if file.endswith('.png'):

        img = cv2.imread(file)
        imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mid = img.shape[0] // 2
        ten_per = int(img.shape[0] * 0.1)
        twenty_per = int(img.shape[0] * 0.2)

        # automatically remove a border of 10% around the image if we have one
        img[0:ten_per, :] = 0
        img[img.shape[0] - ten_per:img.shape[0] - 1, :] = 0
        img[:, 0:ten_per] = 0
        img[:, img.shape[0] - ten_per:img.shape[0] - 1] = 0

        contours, hierarchy = cv2.findContours(imgray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # cv2.drawContours(img, [contours[0]], 0, (0,255,0), 2)

        mainContour = None

        for cnt in contours:

            x, y, w, h = cv2.boundingRect(cnt)
            # print(h/w)
            # cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

            # if h/w > 1 and not ((h/w > 0.90 and h/w < 1.1) or (w/h > 0.90 and w/h < 1.1)) and x < mid - ten_per and x + w > mid + ten_per:
            #     mainContour = cnt
            #     break
            if h / w > 1 and not ((h / w > 0.90 and h / w < 1.1) or (w / h > 0.90 and w / h < 1.1)) and ten_per <= x and \
                    img.shape[0] - ten_per >= x + ten_per:
                mainContour = cnt
                break

        if mainContour is not None:

            x1, y1, w1, h1 = cv2.boundingRect(mainContour)

            # create new image where everything around bounding box is black
            new_img = np.zeros_like(img)
            for j in range(new_img.shape[0]):
                for i in range(new_img.shape[1]):
                    if i >= x1 and i <= x1 + w1 and j >= y1 and j <= y1 + h1:
                        new_img[j][i] = imgray[j][i]

            # print(x,y,w,h)
            # cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

            # if the center is mainly black, then we got here by accident so set to all black
            if np.isclose(img[mid - twenty_per:mid + twenty_per, mid - twenty_per:mid + twenty_per], 0).sum() / img[
                                                                                                                mid - twenty_per:mid + twenty_per,
                                                                                                                mid - twenty_per:mid + twenty_per].size >= 0.90:
                cv2.imwrite(file, np.zeros_like(img))
            else:

                cv2.rectangle(new_img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)

                cv2.imwrite(file, new_img)
        else:
            print(f"No contour found for {file}")
            cv2.imwrite(file, np.zeros_like(img))

        # cv2.imshow('gray', imgray)
        # cv2.imshow('con', img)

        # cv2.waitKey(0)
