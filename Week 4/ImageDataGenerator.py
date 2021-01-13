from tensorflow.keras.preprocessing.image
import ImageDataGenerator

''' instantiate an image generator '''
# pass the rescale to normalize the data
train_datagen = ImageDataGenerator(rescale=1./255) 
# load images from that directory and its sub-directories
# it's a common mistake that people point the generator at the sub-directory. It will fail in that circumtances.
# you should always point it at the directory that contains sub-directories that contain your images.
# The name of the sub-directories will be the labels for you images that are contained within them.

train_generator = train_datagen.flow_from_directory(
    # puit it in the second parameter like this
    train_dir,
    # training a neuron network, all data have to be the same size
    # the images need to be resized to make them consistent
    # The images are resized for you as they're loaded. So, we don't need to preprocess thousands of images on our file system.
    target_size = (300, 300),
    # The images will be loaded for training and validation in batches where it's more efficient than doing it one by one.
    batch_size = 128,
    # This is a binary classifier, because there is only 2 things: horses and human, there will be different class_mode when more than 2 things
    class_mode = 'binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
    # The validation generator should be exactly the same except its points at a different directory, the one containing the sub-directories containing the test images.
    validation_dir,
    target_size = (300, 300),
    batch_size = 32,
    class_mode = 'binary'
)