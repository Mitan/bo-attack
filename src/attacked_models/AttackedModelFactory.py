from attacked_models.mnist.MNISTAttackedModel import MNISTAttackedModel
from enums.DatasetEnum import DatasetEnum


class AttackedModelFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_attacked_model(dataset_descriptor):
        if dataset_descriptor.dataset_type == DatasetEnum.MNIST:
            model = MNISTAttackedModel(weight_load_path=dataset_descriptor.attacked_model_path)
        else:
            raise NotImplementedError
        return model
