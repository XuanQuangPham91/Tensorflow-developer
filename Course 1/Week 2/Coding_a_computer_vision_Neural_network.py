import tensorflow as tf
import nump as np
from tensorflow import keras

# define neural network model
model = keras.Sequential([
    ## model have 3 layers
    # 1st layer is a data shape which match to the images size of 28x28
    # Flatten take a 28x28 square and turns it into a simple linear array
    keras.layers.Flatten(input_shape=(28, 28)),
    # The middle layer (or hidden layer) - the 128 layers can be understood as variables of the functions (x1, x2, ..., xn)
    keras.layers.Dense(128, activation=tf.nn.relu),
    # last model have 10 layers because of 10 classes of clothing in dataset
    keras.layers.Dense(10, activation=tf.nn.softmax)
])