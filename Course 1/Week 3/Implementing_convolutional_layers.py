import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.ops.gen_nn_ops import MaxPool
# from tensorflow import keras
# from tensorflow.python.keras import layers
import nump as np
# from tensorflow import keras

model = tf.keras.models.Sequential([
    ## 1st convolution
    # create 64 filters (which are 3x3 filter)
    # activation is relu: which mean negative value will be through away
    # 1 in the input_shape is the using of 1 kb for color depth, because the image is grayscale
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation='relu',
                           input_shape=(28, 28, 1)),
    # Pooling layer: using MaxPooling for maximum value with size 2x2, then every 4 pixel, the maximum pixel value will be survived
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # We put another Convolution layer for the neural network can learn another set of convolutions on the top of the existing one, and then again pool layer to reduce the size
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax())
])