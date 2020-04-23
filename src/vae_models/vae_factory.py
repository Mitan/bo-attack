from enums.DatasetEnum import DatasetEnum
from vae_models.MNISTVariationalAutoeEncoder import MnistVariationalAutoEncoder


class VAEFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_vae(dataset_descriptor, latent_dimension):
        if dataset_descriptor.dataset_type == DatasetEnum.MNIST:
            vae = MnistVariationalAutoEncoder(dataset_descriptor=dataset_descriptor,
                                              latent_dim=latent_dimension)
        else:
            raise NotImplementedError
        return vae
