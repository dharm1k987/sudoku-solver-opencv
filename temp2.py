import cv2
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical





model = tf.keras.models.load_model('temp2-model')
model.load_weights('temp2-weights.h5')
img = cv2.imread('0-8.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (50, 50))

img = img.reshape(img.shape[0], img.shape[0], 1)
img = np.expand_dims(img, axis=0)

print(img.shape)

print(np.argmax(model.predict(    np.vstack([img])     )) + 1)

exit(0)



model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(50,50,1)),
    tf.keras.layers.Conv2D(64, (2, 2), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (2, 2), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(9, activation="softmax")
])

model.compile('adam', loss='categorical_crossentropy', metrics=['accuracy'])

X = []
y = []

for file in os.listdir('./original'):
    y_val = int(file.split('-')[0]) - 1

    img = cv2.imread('{}/{}'.format('original', file))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (50, 50))
    # print(img.shape)
    X.append(img)
    y.append(y_val)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)

print(X_train.shape, X_test.shape)

y_train = to_categorical(y_train, 9)
y_test = to_categorical(y_test, 9)

model.fit(x=X_train, y=y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test))

model.save('temp2-model')
model.save_weights('temp2-weights.h5')