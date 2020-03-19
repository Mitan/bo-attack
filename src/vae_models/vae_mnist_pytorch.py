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
        return loss

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
                loss_value = self.loss_vae(batch)
                loss_value.backward()
                train_losses.append(loss_value.item())
                if (i + 1) % 10 == 0:
                    print('\rTrain loss:', train_losses[-1],
                          'Batch', i + 1, 'of', total, ' ' * 10, end='', flush=True)
                gd.step()
            test_loss = 0.
            for i, (batch, _) in enumerate(test_loader):
                batch = batch.view(-1, self.original_dim).to(device)
                batch_loss = self.loss_vae(batch)
                test_loss += (batch_loss - test_loss) / (i + 1)
            print('\nTest loss after an epoch: {}'.format(test_loss))


if __name__ == '__main__':
    latent_dim = 2
    batch_size = 100
    vae = MnistVariationalAutoEncoder(latent_dim=latent_dim)
    # vae.train_model(num_epochs=1)

    _, (x_test, y_test) = mnist.load_data()

    image_size = x_test.shape[1]
    original_dim = image_size * image_size
    x_test = np.reshape(x_test, [-1, original_dim])
    x_test = x_test.astype('float32') / 255
    data = (x_test, y_test)

    plot_results(models=[vae.encoder, vae.decoder],
                 data=data,
                 batch_size=batch_size,
                 model_name="vae_mnist_pytorch")

