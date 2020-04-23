# class for storing the parameters of the MNIST dataset
from enums.DatasetEnum import DatasetEnum


class MNISTDescriptor:

    dataset_type = DatasetEnum.MNIST
    # image 2D dimension
    dim = 784
    channels = 1

    # string for printing
    name_string = 'mnist'

    # todo move it from here
    # epsilon used in Attacking the dataset
    epsilon = 0.3

    dataset_folder = './datasets/'
