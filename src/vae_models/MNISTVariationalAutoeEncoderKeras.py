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

code modified by Dmitrii based on the example in keras
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from keras import backend as K
from keras.layers import Lambda, Input, Dense
from keras.losses import binary_crossentropy
from keras.models import Model


class MnistVariationalAutoEncoderKeras:

    def __init__(self, latent_dim, dataset_descriptor):
        self.latent_dim = latent_dim
        self.INTERMEDIATE_DIM = 512
        self.original_dim = dataset_descriptor.dim

        input_shape = (self.original_dim,)

        # VAE model = encoder + decoder

        # build encoder model
        inputs = Input(shape=input_shape, name='encoder_input')
        x_interm = Dense(self.INTERMEDIATE_DIM, activation='relu')(inputs)
        x = Dense(self.INTERMEDIATE_DIM, activation='relu')(x_interm)
        z_mean = Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = Dense(self.latent_dim, name='z_log_var')(x)

        # use reparameterization trick to push the sampling out as input
        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self._sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])

        # instantiate encoder model
        encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
        self.encoder = encoder

        # build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x_interm = Dense(self.INTERMEDIATE_DIM, activation='relu')(latent_inputs)
        x = Dense(self.INTERMEDIATE_DIM, activation='relu')(x_interm)
        outputs = Dense(self.original_dim, activation='sigmoid')(x)

        # instantiate decoder model
        decoder = Model(latent_inputs, outputs, name='decoder')
        self.decoder = decoder

        # instantiate VAE model
        outputs = decoder(encoder(inputs)[2])

        vae = Model(inputs, outputs, name='vae_mlp')

        # compute loss
        reconstruction_loss = binary_crossentropy(inputs, outputs)
        reconstruction_loss *= self.original_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)

        vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        self.vae = vae

    # reparameterization trick
    # instead of sampling from Q(z|X), sample epsilon = N(0,I)
    # z = z_mean + sqrt(var) * epsilon
    @staticmethod
    def _sampling(args):
        """ Reparameterization trick by sampling from an isotropic unit Gaussian.
        # Arguments

            args (tensor): mean and log of variance of Q(z|X)
        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def train(self, data_loader, weights=None, batch_size=128, epochs=50):

        x_train = np.vstack([data_loader.train_data, data_loader.validation_data])
        # using test data here is a bit strange, but this is what the original keras example did
        x_test = data_loader.test_data

        if weights:
            self.vae.load_weights(weights)
        else:
            self.vae.fit(x_train,
                         epochs=epochs,
                         batch_size=batch_size,
                         validation_data=(x_test, None))
            self.vae.save_weights('vae_mlp_mnist.h5')

    # encode input or a batch of inputs
    # expects a 2D numpy array of shape (n_points * original_dim)
    def encode(self, inputs):
        outputs, _, _ = self.encoder.predict(inputs)
        return outputs

    # encode input or a batch of inputs
    # expects a 2D numpy array of shape (n_points * latent_dim)
    def decode(self, inputs):
        return self.decoder.predict(inputs)
