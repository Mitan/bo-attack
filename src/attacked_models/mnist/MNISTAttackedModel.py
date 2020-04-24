from tensorflow.contrib.keras.api.keras.layers import Conv2D, MaxPooling2D
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten
from tensorflow.contrib.keras.api.keras.models import Sequential


class MNISTAttackedModel:
    def __init__(self, weight_load_path=None, use_softmax=False):
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

    def predict(self, data):
        # data = data.reshape(28,28,1)
        return self.model(data.reshape(28,28,1))