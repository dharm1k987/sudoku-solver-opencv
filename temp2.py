import cv2
import numpy as np
import tensorflow as tf
import time

model = tf.keras.models.load_model('model.hdf5')


img1 = cv2.imread('imgs/sudoku.jpg')
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img1 = img1.reshape(img1.shape[0], img1.shape[0], 1)
# img1 = np.expand_dims(img1, axis=0)


start = time.time()
mainList = []
for i in range(0,3):
    img1 = cv2.resize(img1.copy(), (32, 32))
    mainList.append(img1)
# img2 = np.expand_dims(img2, axis=0)


img = np.array(mainList)

ans = model(    tf.convert_to_tensor(img)     )
print(list(map(np.argmax, ans)))
print('{}s'.format(time.time()-start))


start = time.time()
for i in mainList:
    i = np.expand_dims(i, axis=0)
    np.argmax(model(i, training=False)) + 1
print('{}s'.format(time.time()-start))


# cv2.imwrite('{}.png'.format(i), img)
#
# img = img.reshape(img.shape[0], img.shape[0], 1)
# img = np.expand_dims(img, axis=0)
# pred = np.argmax(model(img, training=False)) + 1