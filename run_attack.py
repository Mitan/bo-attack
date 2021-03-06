from src.bo.AttackRunner import AttackRunner
from src.dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from src.dataset_data.mnist.MnistLoader import MnistLoader
from src.objective_func.ObjectiveFunctionEvaluator import ObjectiveFunctionEvaluator
from src.utilities.InitialDataGenerator import InitialDataGenerator

if __name__ == '__main__':
    dataset_descriptor = MNISTDescriptor()

    mnist_loader = MnistLoader()
    mnist_loader.load_data(dataset_folder=dataset_descriptor.dataset_folder)
    test_image_0 = mnist_loader.test_data[1, :]
    evaluator = ObjectiveFunctionEvaluator(dataset_descriptor=dataset_descriptor,
                                           use_softmax=True,
                                           rescale=True,
                                           target_image=test_image_0,
                                           target_class=0)
    # print(evaluator.attacked_model.predict(test_image_0))
    # print(evaluator.evaluate(0))

    attack_runner = AttackRunner(objective_function_evaluator=evaluator,
                                 dataset_descriptor=dataset_descriptor,
                                 domain_dimensions=dataset_descriptor.bo_dimensions)

    initial_data_generator = InitialDataGenerator(dataset_descriptor=dataset_descriptor,
                                                  objective_function_evaluator=evaluator)

    initial_inputs, initial_outputs = initial_data_generator.generate(num_images=5)

    attack_runner.init_bo(initial_dimensions=dataset_descriptor.initial_dimensions,
                          num_initial_observations=dataset_descriptor.initial_observations,
                          initial_history_inputs=initial_inputs,
                          initial_history_outputs=initial_outputs)

    attack_runner.run(bos_iterations=dataset_descriptor.bos_iterations,
                      total_iterations_max=dataset_descriptor.total_iterations)
