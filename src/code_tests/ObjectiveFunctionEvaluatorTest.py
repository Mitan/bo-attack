from attacked_models.AttackedModelFactory import AttackedModelFactory
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from dataset_data.mnist.MnistLoader import MnistLoader
import numpy as np

from objective_func.ObjectiveFunctionEvaluator import ObjectiveFunctionEvaluator

if __name__ == '__main__':
    latent_dim = 32
    mnist_root = '../../datasets/'

    dataset_descriptor = MNISTDescriptor()
    dataset_descriptor.dataset_folder = '../../datasets/'

    mnist_loader = MnistLoader()
    mnist_loader.load_data(dataset_folder=dataset_descriptor.dataset_folder)

    dataset_descriptor.attacked_model_path = '../attacked_models/mnist/mnist'
    attacked_model = AttackedModelFactory.get_attacked_model(dataset_descriptor)

    # best_class_scores = np.argmin(attacked_model.predict(mnist_loader.test_data).max(axis=1))
    # print(best_class_scores)

    # index of the most uncertain image - the smallest max score
    image_0_index = 6572

    test_image_0 = mnist_loader.test_data[6572, :]
    image_0_scores = attacked_model.predict(test_image_0)
    # print(np.sort(image_0_scores))
    # print(image_1_scores)
    image_0_class = image_0_scores.argmax()
    # print(image_0_class, mnist_loader.test_labels[image_0_index].argmax())

    test_image_1 = mnist_loader.test_data[1, :]
    image_1_scores = attacked_model.predict(test_image_1)
    print(image_1_scores)
    image_1_class = image_1_scores.argmax()
    print(image_0_class, image_1_class)

    dataset_descriptor.epsilon = 1.0
    evaluator = ObjectiveFunctionEvaluator(dataset_descriptor=dataset_descriptor,
                                           use_softmax=True,
                                           rescale=False,
                                           target_image=test_image_0,
                                           target_class=None)

    perturbation = - (test_image_0 - test_image_1)
    print(evaluator.evaluate(perturbation))






