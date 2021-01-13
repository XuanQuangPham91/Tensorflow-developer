import tensorflow as tf
from os import path, getcwd, chdir

path = f"{getcwd()}/../tmp2/mnist.npz"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


# GRADED FUNCTION: train_mnist_conv
def train_mnist_conv():
    # Please write your code only where you are indicated.
    # please do not remove model fitting inline comments.

    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE

    mnist = tf.keras.datasets.mnist
    (training_images,
     training_labels), (test_images, test_labels) = mnist.load_data(path=path)
    # YOUR CODE STARTS HERE
    training_images = training_images.reshape(60000, 28, 28, 1)
    training_images = training_images / 255
    test_images = test_images.re
    test_images = test_images / 255
    # YOUR CODE ENDS HERE

    model = tf.keras.models.Sequential([
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # model fitting
    history = model.fit(
        # YOUR CODE STARTS HERE

        # YOUR CODE ENDS HERE
    )
    # model fitting
    return history.epoch, history.history['acc'][-1]
