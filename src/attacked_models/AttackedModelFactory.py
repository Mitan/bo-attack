from src.attacked_models.mnist.MNISTAttackedModel import MNISTAttackedModel
from src.enums.DatasetEnum import DatasetEnum


class AttackedModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_attacked_model(dataset_descriptor, use_softmax=True):
        if dataset_descriptor.dataset_type == DatasetEnum.MNIST:
            model = MNISTAttackedModel(weight_load_path=dataset_descriptor.attacked_model_path,
                                       use_softmax=use_softmax)
        else:
            raise NotImplementedError
        return model
