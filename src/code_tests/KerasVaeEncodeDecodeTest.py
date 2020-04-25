import matplotlib.pylab as plt

import numpy as np

from attacked_models.AttackedModelFactory import AttackedModelFactory
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from dataset_data.mnist.MnistLoader import MnistLoader
from vae_models.MNISTVariationalAutoeEncoderKeras import MnistVariationalAutoEncoderKeras


def plot(num_figures, original_figures, transformed_figures):
    digit_size = 28
    figure = np.zeros((digit_size * num_figures, digit_size * 2))
    figure[:, :digit_size] = original_figures
    figure[:, digit_size:] = transformed_figures

    plt.imshow(figure, cmap='Greys_r')
    # plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    latent_dim = 32
    mnist_root = '../../datasets/'

    dataset_descriptor = MNISTDescriptor()
    dataset_descriptor.dataset_folder = '../../datasets/'

    vae = MnistVariationalAutoEncoderKeras(latent_dim=latent_dim,
                                           dataset_descriptor=dataset_descriptor)
    mnist_loader = MnistLoader()
    mnist_loader.load_data(dataset_folder=dataset_descriptor.dataset_folder)

    vae.train(data_loader=mnist_loader, epochs=100)

    # x = np.atleast_2d([[0.1, 0.9, 0.9], [0.1, 0.9, 0.9]])
    # print(x.shape)
    # predicted_x = vae.decode(x)
    # print(predicted_x.shape)
    # x_back = vae.encode(predicted_x)
    # print(x_back.shape)

    x = np.atleast_2d([np.ones(latent_dim), np.zeros(latent_dim)])
    y = vae.decode(x)

    x_back = vae.encode(y)
    # print(x_back)
    num_images = 20
    test_images = mnist_loader.test_data[:num_images, :]

    dataset_descriptor.attacked_model_path = '../attacked_models/mnist/mnist'
    attacked_model = AttackedModelFactory.get_attacked_model(dataset_descriptor)

    original_prediction = attacked_model.predict(test_images)
    original_predcited_classes = original_prediction.argmax(axis=1)
    # print(original_predcited_classes)

    transformed_images = vae.decode(vae.encode(test_images))
    transformed_prediction = attacked_model.predict(transformed_images)
    transformed_predcited_classes = transformed_prediction.argmax(axis=1)

    plot(num_figures=num_images,
         original_figures=test_images.reshape(-1, 28),
         transformed_figures=transformed_images.reshape(-1, 28))
    # transformed_predcited_classes = mnist_loader.test_labels[:num_images, :].argmax(axis=1)
    # print(transformed_predcited_classes)


