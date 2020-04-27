"""
A class to generate  random initial images for performing BO during attack
"""
import numpy as np
from pyDOE import lhs


class InitialDataGenerator:
    def __init__(self, dataset_descriptor, objective_function_evaluator):
        """
        :type objective_function_evaluator: evaluator of the objective function (the score of the perturbed image).
        :type dataset_descriptor: the descriptor of the dataset.
        """
        self.objective_function_evaluator = objective_function_evaluator
        self.dataset_descriptor = dataset_descriptor

    # generate the required number of images
    def generate(self, num_images, method='lhs'):
        noise_var = 1.0e-10
        bounds = self.dataset_descriptor.bo_bounds
        d = bounds.shape[0]

        if method == 'lhs':
            x_init = lhs(d, num_images) * (bounds[0, 1] - bounds[0, 0]) + bounds[0, 0]
        else:
            x_init = np.random.uniform(low=bounds[0, 0], high=bounds[0, 1], size=(num_images, d))
        x_init = x_init.astype("float32")

        f_init = np.apply_along_axis(func1d=self.objective_function_evaluator.evaluate,
                                     axis=1,
                                     arr=x_init).reshape(-1, 1)

        y_init = f_init + np.sqrt(noise_var) * np.random.randn(num_images, 1)
        return x_init, y_init
