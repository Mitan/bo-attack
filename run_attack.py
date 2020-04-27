from bo.AttackRunner import AttackRunner
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from dataset_data.mnist.MnistLoader import MnistLoader
from objective_func.ObjectiveFunctionEvaluator import ObjectiveFunctionEvaluator

if __name__ == '__main__':
    dataset_descriptor = MNISTDescriptor()

    mnist_loader = MnistLoader()
    mnist_loader.load_data(dataset_folder=dataset_descriptor.dataset_folder)
    test_image_0 = mnist_loader.test_data[0, :]
    evaluator = ObjectiveFunctionEvaluator(dataset_descriptor=dataset_descriptor,
                                           use_softmax=True,
                                           rescale=False,
                                           target_image=test_image_0,
                                           target_class=None)

    attack_runner = AttackRunner(objective_function_evaluator=None,
                                 dataset_descriptor=dataset_descriptor,
                                 domain_dimensions=dataset_descriptor.bo_dimensions)

    attack_runner.init_bo()