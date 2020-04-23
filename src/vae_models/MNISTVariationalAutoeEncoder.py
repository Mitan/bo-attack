"""Example of VAE on MNIST dataset using MLP
The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean = 0 and std = 1.
# Reference
[1] Kingma, Diederik P., and Max Welling.
"Auto-Encoding Variational Bayes."
https://arxiv.org/abs/1312.6114

code modified by Dmitrii
"""
from itertools import chain

import torch
from torch import nn
from torch.distributions import Normal, Bernoulli, Independent
from torchvision import transforms
from torchvision.datasets import MNIST


class MnistVariationalAutoEncoder:

    def __init__(self, latent_dim, dataset_descriptor):
        # run some pytorch init code

        torch.manual_seed(0)

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        self.device = device

        print('Using torch version {}'.format(torch.__version__))
        print('Using {} device'.format(device))

        # note: hardcoded
        self.INTERMEDIATE_DIM = 100

        # note: hardcoded
        self.batch_size = 100
        # note: hardcoded
        self.learning_rate = 1e-3

        self.latent_dim = latent_dim
        self.original_dim = dataset_descriptor.dim

        encoder = nn.Sequential(
            nn.Linear(self.original_dim, self.INTERMEDIATE_DIM),
            nn.ReLU(),
            nn.Linear(self.INTERMEDIATE_DIM, self.INTERMEDIATE_DIM),
            nn.ReLU(),
            nn.Linear(self.INTERMEDIATE_DIM, 2 * latent_dim))  # note that the final layer outputs real values

        decoder = nn.Sequential(
            nn.Linear(latent_dim, self.INTERMEDIATE_DIM),
            nn.ReLU(),
            nn.Linear(self.INTERMEDIATE_DIM, self.INTERMEDIATE_DIM),
            nn.ReLU(),
            nn.Linear(self.INTERMEDIATE_DIM, self.original_dim))

        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

    def _loss_vae(self, x):

        batch_size = x.size(0)
        encoder_output = self.encoder(x)
        pz = Independent(Normal(loc=torch.zeros(batch_size, self.latent_dim).to(self.device),
                                scale=torch.ones(batch_size, self.latent_dim).to(self.device)),
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

    def train(self, num_epochs, dataset_folder=None):

        # Training dataset
        download = dataset_folder is None
        train_loader = torch.utils.data.DataLoader(
            MNIST(root=dataset_folder, train=True, download=download,
                  transform=transforms.ToTensor()),
            batch_size=self.batch_size, shuffle=True, pin_memory=True)

        gd = torch.optim.Adam(
            chain(*[x.parameters() for x in [self.encoder, self.decoder]
                    if (isinstance(x, nn.Module) or isinstance(x, nn.Parameter))]),
            lr=self.learning_rate)
        train_losses = []
        for ep in range(num_epochs):
            for i, (batch, _) in enumerate(train_loader):
                total = len(train_loader)
                gd.zero_grad()
                batch = batch.view(-1, self.original_dim).to(self.device)
                loss_value, _ = self._loss_vae(batch)
                loss_value.backward()
                train_losses.append(loss_value.item())
                if (i + 1) % 10 == 0:
                    print('\rTrain loss: {}. Batch {} of {} for epoch {}'.
                          format(train_losses[-1], i + 1, total, ep), end='', flush=True)
                gd.step()

    # encode input or a batch of inputs
    # expects a 2D numpy array of shape (n_points * original_dim)
    def encode(self, inputs):
        with torch.no_grad():
            transformed_input = torch.tensor(inputs).float().view(-1, self.original_dim)
            # take only mean
            x_encoded = self.encoder(transformed_input).numpy().T[:self.latent_dim].T
        return x_encoded

    # encode input or a batch of inputs
    # expects a 2D numpy array of shape (n_points * latent_dim)
    def decode(self, inputs):
        with torch.no_grad():
            transformed_input = torch.tensor(inputs).float().view(-1, self.latent_dim)
            x_decoded = torch.sigmoid(self.decoder(transformed_input)).numpy()
        return x_decoded
