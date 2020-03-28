# class for storing the parameters of the MNIST dataset


class MNISTDescriptor:
    # image 2D dimension
    dim = 784
    channels = 1

    # string for printing
    name_string = 'mnist'

    # todo move it from here
    # epsilon used in Attacking the dataset
    epsilon = 0.3
