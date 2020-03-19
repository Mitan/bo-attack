'''Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114

code modified by Dmitrii based on the example in keras
'''
import os
from itertools import chain

from torch.distributions import Normal, Bernoulli, Independent

from torchvision.datasets import MNIST
from torchvision import transforms
import torch
from torch import nn
import numpy as np
import matplotlib.pylab as plt
from keras.datasets import mnist

torch.manual_seed(0)

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

print('Using torch version {}'.format(torch.__version__))
print('Using {} device'.format(device))


def plot_results(vae,
                 test_loader,
                original_dim,
                 latent_dim,
                 batch_size=128,
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
        batch = batch.view(-1, original_dim ).to(device)
        z_mean = encoder(batch)[:, :latent_dim]
        y_test = test_loader.dataset.targets
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    # plt.savefig(filename)
    # plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
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


class MnistVariationalAutoEncoder(object):

    def __init__(self, latent_dim, original_dim=784, intermediate_dim=100):
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.original_dim = original_dim

        encoder = nn.Sequential(
            nn.Linear(original_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, 2 * latent_dim))  # note that the final layer outputs real values

        decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, original_dim))

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def loss_vae(self, x):

        batch_size = x.size(0)
        encoder_output = self.encoder(x)
        pz = Independent(Normal(loc=torch.zeros(batch_size, self.latent_dim).to(device),
                                scale=torch.ones(batch_size, self.latent_dim).to(device)),
                         reinterpreted_batch_ndims=1)
        qz_x = Independent(Normal(loc=encoder_output[:, :self.latent_dim],
                                  scale=torch.exp(encoder_output[:, self.latent_dim:])),
                           reinterpreted_batch_ndims=1)

        z = qz_x.rsample()

        decoder_output = self.decoder(z)
        px_z = Independent(Bernoulli(logits=decoder_output),
                           reinterpreted_batch_ndims=1)
        loss = -(px_z.log_prob(x) + pz.log_prob(z) - qz_x.log_prob(z)).mean()
        return loss, decoder_output

    def train_model(self, train_loader=None, test_loader=None, batch_size=100, num_epochs=3, learning_rate=1e-3):
        if not train_loader:
            # Training dataset
            train_loader = torch.utils.data.DataLoader(
                MNIST(root='.', train=True, download=True,
                      transform=transforms.ToTensor()),
                batch_size=batch_size, shuffle=True, pin_memory=True)

        if not test_loader:
            # Test dataset
            test_loader = torch.utils.data.DataLoader(
                MNIST(root='.', train=False, transform=transforms.ToTensor()),
                batch_size=batch_size, shuffle=True, pin_memory=True)

        model = [self.encoder, self.decoder]
        gd = torch.optim.Adam(
            chain(*[x.parameters() for x in model
                    if (isinstance(x, nn.Module) or isinstance(x, nn.Parameter))]),
            lr=learning_rate)
        train_losses = []
        for _ in range(num_epochs):
            for i, (batch, _) in enumerate(train_loader):
                total = len(train_loader)
                gd.zero_grad()
                batch = batch.view(-1, self.original_dim).to(device)
                loss_value, _ = self.loss_vae(batch)
                loss_value.backward()
                train_losses.append(loss_value.item())
                if (i + 1) % 10 == 0:
                    print('\rTrain loss:', train_losses[-1],
                          'Batch', i + 1, 'of', total, ' ' * 10, end='', flush=True)
                gd.step()
            test_loss = 0.
            for i, (batch, _) in enumerate(test_loader):
                batch = batch.view(-1, self.original_dim).to(device)
                batch_loss, _ = self.loss_vae(batch)
                test_loss += (batch_loss - test_loss) / (i + 1)
            print('\nTest loss after an epoch: {}'.format(test_loss))


if __name__ == '__main__':
    latent_dim = 2
    batch_size = 100
    vae = MnistVariationalAutoEncoder(latent_dim=latent_dim)
    vae.train_model(num_epochs=16)

    test_loader = torch.utils.data.DataLoader(
        MNIST(root='.', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, pin_memory=True)

    plot_results(vae=vae,
                 test_loader=test_loader,
                 original_dim=784,
                 latent_dim=latent_dim,
                 batch_size=batch_size,
                 model_name="vae_mnist_pytorch")

