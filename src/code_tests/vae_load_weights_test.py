from attacked_models.AttackedModelFactory import AttackedModelFactory
from dataset_data.mnist.MnistLoader import MnistLoader
from src.dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from vae_models.MNISTVariationalAutoeEncoderPytorch import MnistVariationalAutoEncoderPytorch

import numpy as np


def train_one_vae(weights_root_folder, dataset_descriptor, latent_dim):
    vae = MnistVariationalAutoEncoderPytorch(dataset_descriptor=dataset_descriptor,
                                             latent_dim=latent_dim)
    vae.train(num_epochs=dataset_descriptor.vae_num_epochs,
              dataset_folder=dataset_descriptor.dataset_folder)
    vae.save_weights(save_folder=weights_root_folder)
    return vae


if __name__ == '__main__':
    mnist_descriptor = MNISTDescriptor()
    mnist_descriptor.vae_num_epochs = 1
    weights_folder = '../vae_models/vae_weights/'
    mnist_descriptor.dataset_folder = '../../datasets/'

    dim = 3
    trained_vae = train_one_vae(weights_root_folder=weights_folder,
                                dataset_descriptor=mnist_descriptor,
                                latent_dim=dim)

    mnist_loader = MnistLoader()
    mnist_loader.load_data(dataset_folder=mnist_descriptor.dataset_folder)

    test_image = mnist_loader.test_data[0, :]

    mnist_descriptor.attacked_model_path = '../attacked_models/mnist/mnist'
    attacked_model = AttackedModelFactory.get_attacked_model(mnist_descriptor)

    trained_vae_predictions = attacked_model.predict(trained_vae.decode(trained_vae.encode(test_image)))
    print(trained_vae_predictions)

    untrained_vae = MnistVariationalAutoEncoderPytorch(dataset_descriptor=mnist_descriptor,
                                                       latent_dim=dim)
    untrained_vae.load_weights(load_folder=weights_folder,
                               num_epochs_trained=mnist_descriptor.vae_num_epochs)

    untrained_vae_predictions = attacked_model.predict(untrained_vae.decode(trained_vae.encode(test_image)))
    print(untrained_vae_predictions)
    assert np.allclose(untrained_vae_predictions, trained_vae_predictions)
