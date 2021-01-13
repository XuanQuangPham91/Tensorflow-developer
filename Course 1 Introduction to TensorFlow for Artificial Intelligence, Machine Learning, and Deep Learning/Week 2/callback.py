import tensorflow as tf
import numpy as np
from tensorflow.keras import callbacks

# from tensorflow import keras


# callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        # use acc (instead of accuracy) for old version
        if (logs.get('accuracy') > 0.99):
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist
(training_images, training_lables), (test_images,
                                     test_lables) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

## training stage
# Use the callbacks parameter, and pass it in this instance of the class
model.fit(training_images, training_lables, epochs=10, callbacks=[callbacks])
