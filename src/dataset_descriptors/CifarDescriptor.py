# class for storing the parameters of the CIFAR-10 dataset
from enums.DatasetEnum import DatasetEnum


class CIFARDescriptor:

    dataset_type = DatasetEnum.CIFAR10
    # image 2D dimension
    high_dim = int(32 * 32)
    channels = 3

    # string for printing
    name_string = 'cifar'

    # todo move it from here
    # epsilon used in Attacking the dataset
    epsilon = 0.05
