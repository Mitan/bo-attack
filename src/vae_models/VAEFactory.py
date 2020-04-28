from src.enums.DatasetEnum import DatasetEnum
from src.vae_models.MNISTVariationalAutoeEncoderPytorch import MnistVariationalAutoEncoderPytorch


class VAEFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_vae(dataset_descriptor, latent_dimension):
        if dataset_descriptor.dataset_type == DatasetEnum.MNIST:
            vae = MnistVariationalAutoEncoderPytorch(dataset_descriptor=dataset_descriptor,
                                                     latent_dim=latent_dimension)
        else:
            raise NotImplementedError
        return vae
