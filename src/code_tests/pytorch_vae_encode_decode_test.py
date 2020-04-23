import numpy as np

from dataset_descriptors.MnistDescriptor import MNISTDescriptor
from vae_models.VAEFactory import VAEFactory

if __name__ == '__main__':
    latent_dim = 10
    mnist_root = '../../datasets/'

    dataset_descriptor = MNISTDescriptor()
    vae = VAEFactory.get_vae(dataset_descriptor=dataset_descriptor,
                             latent_dimension=latent_dim)
    vae.train(num_epochs=50, dataset_folder=mnist_root)

    """
    batch_size = vae.batch_size
    test_loader = torch.utils.data.DataLoader(
        MNIST(root=mnist_root, train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True, pin_memory=True)   
    """
    # x = np.atleast_2d([0.1, 0.9, -0.5, 0.7])

    x = np.atleast_2d([np.ones(latent_dim), np.zeros(latent_dim)])
    # print(x.shape)
    y = vae.decode(x)
    # print(y.shape)

    x_back = vae.encode(y)
    print(x_back)
