# class for storing the parameters of the MNIST dataset
from enums.AcquisitionEnum import AcquisitionEnum
from enums.DatasetEnum import DatasetEnum
from enums.GPEnum import GPEnum


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

    attacked_model_path = './src/attacked_models/mnist/mnist'

    # todo move to a separate BO descriptor class
    # the bound for the image pixels after rescaling from [0, 255] range
    image_bounds = [0,1]

    acq_type = AcquisitionEnum.AdditiveLCB

    num_subspace = 12

    # the bound on summary statistic for BOS
    y_bounds_bos = [-1, 1]

    gp_type = GPEnum.AdditiveGP

    # the frequency of re-learning the GP hypers
    gp_update_freq = 5
