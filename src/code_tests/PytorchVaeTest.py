import matplotlib.pylab as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import MNIST

from attacked_models.AttackedModelFactory import AttackedModelFactory
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from vae_models.MNISTVariationalAutoeEncoderPytorch import MnistVariationalAutoEncoderPytorch
from vae_models.VAEFactory import VAEFactory


def plot_results(vae,
                 test_loader,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder = vae.encoder
    decoder = vae.decoder
    with torch.no_grad():
        batch = (test_loader.dataset.data.float() / 255.)
        print(batch.numpy().shape)
        batch = batch.view(-1, vae.original_dim).to(vae.device)
        z_mean = encoder(batch)[:, :latent_dim]
        y_test = test_loader.dataset.targets
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    # plt.savefig(filename)
    # plt.show()

    filename = "./{}_digits_over_latent.png".format(model_name)
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            with torch.no_grad():
                z_sample = torch.tensor(z_sample).float().view(-1, latent_dim)
                x_decoded = torch.sigmoid(decoder(z_sample))
                digit = x_decoded.reshape(digit_size, digit_size)
                figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = (n - 1) * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


def plot(num_figures, original_figures, transformed_figures):
    digit_size = 28
    figure = np.zeros((digit_size * num_figures, digit_size * 3))
    figure[:, :digit_size] = original_figures
    figure[:, digit_size: 2 * digit_size] = transformed_figures
    figure[:, 2 * digit_size: ] = transformed_figures - original_figures

    plt.imshow(figure, cmap='Greys_r')
    # plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    latent_dim = 2
    mnist_root = '../../datasets/'

    dataset_descriptor = MNISTDescriptor()
    vae = VAEFactory.get_vae(latent_dimension=latent_dim, dataset_descriptor=dataset_descriptor)
    vae.train(num_epochs=15, dataset_folder=mnist_root)

    batch_size = vae.batch_size
    test_loader = torch.utils.data.DataLoader(
        MNIST(root=mnist_root, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    plot_results(vae=vae,
                 test_loader=test_loader,
                 model_name="vae_mnist_pytorch")
    """
    num_points = 10
    batch = (test_loader.dataset.data.float() / 255.)

    batch = batch[: num_points, :, :].view(-1, vae.original_dim).to(vae.device)
    original_inputs = batch.numpy()

    encoded_x = vae.encode(original_inputs)
    transformed_inputs =  vae.decode(encoded_x)

    plot(num_figures=num_points,
         original_figures=original_inputs.reshape(-1, 28),
         transformed_figures=transformed_inputs.reshape(-1, 28))

    dataset_descriptor.attacked_model_path = '../attacked_models/mnist/mnist'
    attacked_model = AttackedModelFactory.get_attacked_model(dataset_descriptor)

    original_prediction = attacked_model.predict(original_inputs)
    original_predcited_classes = original_prediction.argmax(axis=1)

    transformed_prediction = attacked_model.predict(transformed_inputs)
    transformed_predcited_classes = transformed_prediction.argmax(axis=1)
    print(original_predcited_classes)
    print(transformed_predcited_classes)
    """
