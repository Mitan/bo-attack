import numpy as np

from attacked_models.AttackedModelFactory import AttackedModelFactory
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from vae_models.VAEFactory import VAEFactory

if __name__ == '__main__':
    latent_dim = 10
    mnist_root = '../../datasets/'

    dataset_descriptor = MNISTDescriptor()
    vae = VAEFactory.get_vae(dataset_descriptor=dataset_descriptor,
                             latent_dimension=latent_dim)
    # vae.train(num_epochs=50, dataset_folder=mnist_root)

    x = np.atleast_2d(np.ones(latent_dim))
    y = vae.decode(x)
    print(y.shape)

    dataset_descriptor.attacked_model_path = '../src/attacked_models/mnist/mnist'
    attacked_model = AttackedModelFactory.get_attacked_model(dataset_descriptor)
    predictions = attacked_model.predict(y)
    print(predictions)