import time

import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from models import model


def model_wrapper(wts_path, train=False, to_save_as=False, model_path=None):
    if model_path:
        return tf.keras.models.load_model(model_path)

    my_model = model.get_model()

    if wts_path:
        my_model.load_weights(wts_path)

    if train:
        class myCallback(Callback):
            def on_epoch_end(self, epoch, logs={}):
                if logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95:
                    print('Stopping training')
                    my_model.stop_training = True

        callbacks = myCallback()

        mnist = tf.keras.datasets.mnist

        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        # normalize the data
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        my_model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
        print(my_model.evaluate(x_test, y_test))

        if wts_path:
            my_model.save_weights('{}-{}'.format(wts_path, round(time.time())))
        else:
            my_model.save_weights(to_save_as)

    return my_model
