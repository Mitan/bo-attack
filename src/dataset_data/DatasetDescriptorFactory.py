from dataset_data.cifar.CifarDescriptor import CIFARDescriptor
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from enums.DatasetEnum import DatasetEnum


class DatasetDescriptionFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset_descriptor(dataset_type):

        if dataset_type == DatasetEnum.MNIST:
            descriptor = MNISTDescriptor()
        elif dataset_type == DatasetEnum.CIFAR10:
            descriptor = CIFARDescriptor()

        else:
            raise NotImplementedError
        return descriptor
