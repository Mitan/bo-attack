from src.dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from vae_models.MNISTVariationalAutoeEncoderPytorch import MnistVariationalAutoEncoderPytorch


def train_one_vae(weights_root_folder, dataset_descriptor, latent_dim):
    vae = MnistVariationalAutoEncoderPytorch(dataset_descriptor=dataset_descriptor,
                                             latent_dim=latent_dim)
    vae.train(num_epochs=dataset_descriptor.vae_num_epochs,
              dataset_folder=dataset_descriptor.dataset_folder)
    vae.save_weights(save_folder=weights_root_folder)


if __name__ == '__main__':
    mnist_descriptor = MNISTDescriptor()
    # mnist_descriptor.vae_num_epochs = 1
    weights_folder = './src/vae_models/vae_weights/'

    for dim in mnist_descriptor.bo_dimensions:
        train_one_vae(weights_root_folder=weights_folder,
                      dataset_descriptor=mnist_descriptor,
                      latent_dim=dim)
