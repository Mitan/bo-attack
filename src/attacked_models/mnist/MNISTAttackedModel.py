"""
This code is modified by Dmitrii from the original code by Nicholas Carlini <nicholas@carlini.com>.
"""

import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Activation, Flatten
from keras.models import Sequential


# from keras.models import Sequential
# from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense


class MNISTAttackedModel:
    # note we are using softmax to get bounds [-1, 1] bounds for the BO objective function
    def __init__(self,  use_softmax, weight_load_path=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        model = Sequential()

        model.add(Conv2D(32, (3, 3),
                         input_shape=(28, 28, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(200))
        model.add(Activation('relu'))
        model.add(Dense(10))
        # output log probability, used for black-box attack
        if use_softmax:
            model.add(Activation('softmax'))
        if weight_load_path:
            model.load_weights(weight_load_path)

        self.model = model

    # the input point should be a np array and have dimensions (num_points x 784)
    def predict(self, data):
        """
        the original code by Nicholas Carlini assumed MNIST normalisation to be x-> x/255 - 0.5.
        But in this case there are some instabilities in training the VAE with this normalisation
        So here we assume the normalisation x -> x/255,
        and then subtract 0.5 here to match the normalisation from Nicholas Carlini

        """
        return K.eval(self.model((data - 0.5).reshape(-1, 28,28,1)))