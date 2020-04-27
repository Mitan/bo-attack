# class for storing the parameters of the MNIST dataset
from enums.AcquisitionEnum import AcquisitionEnum
from enums.DatasetEnum import DatasetEnum
from enums.GPEnum import GPEnum
import numpy as np

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

    num_subspace = 12

    # the bound on summary statistic for BOS
    y_bounds_bos = [-1, 1]

    gp_type = GPEnum.AdditiveGP
    acq_type = AcquisitionEnum.AdditiveLCB

    gp_type = GPEnum.SimpleGP
    acq_type = AcquisitionEnum.LCB

    # the frequency of re-learning the GP hypers
    gp_update_freq = 5
    # set it similar to BayesOpt attack code
    bo_dimensions = [i**2 for i in range(6, 21, 2)]

    initial_dimensions = [6**2, 20**2, 14**2]

    initial_observations = 30

    total_observations = 900

    vae_num_epochs = 50


