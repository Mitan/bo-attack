from bo.AttackRunner import AttackRunner
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from dataset_data.mnist.MnistLoader import MnistLoader
from objective_func.ObjectiveFunctionEvaluator import ObjectiveFunctionEvaluator
from utilities.InitialDataGenerator import InitialDataGenerator

if __name__ == '__main__':
    dataset_descriptor = MNISTDescriptor()

    mnist_loader = MnistLoader()
    mnist_loader.load_data(dataset_folder=dataset_descriptor.dataset_folder)
    test_image_0 = mnist_loader.test_data[0, :]
    evaluator = ObjectiveFunctionEvaluator(dataset_descriptor=dataset_descriptor,
                                           use_softmax=True,
                                           rescale=False,
                                           target_image=test_image_0,
                                           target_class=0)
    # print(evaluator.attacked_model.predict(test_image_0))
    # print(evaluator.evaluate(0))

    attack_runner = AttackRunner(objective_function_evaluator=evaluator,
                                 dataset_descriptor=dataset_descriptor,
                                 domain_dimensions=dataset_descriptor.bo_dimensions)

    initial_data_generator = InitialDataGenerator(dataset_descriptor=dataset_descriptor,
                                                  objective_function_evaluator=evaluator)

    initial_inputs, initial_outputs = initial_data_generator.generate(num_images=5, method='a')

    attack_runner.init_bo(initial_dimensions=dataset_descriptor.initial_dimensions,
                          num_initial_observations=dataset_descriptor.initial_observations)
