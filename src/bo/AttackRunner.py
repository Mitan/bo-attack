"""
A class for runner the outer BO loop on dimensions
"""
from bo.DimensionBORunner import DimensionBORunner
import numpy as np

# the paper uses 30 initial evaluations
# 900 max iterations
# GP update frequency of 5 iterations
# learning the dimension with frequency 8 * update_for_hypers, i.e. 8 * 5
from bo.FixedDimensionRunner import FixedDimensionRunner


class AttackRunner:
    def __init__(self, domain_dimensions, dataset_descriptor, objective_function_evaluator):
        """
        :type objective_function_evaluator: evaluator of the objective function (the score of the perturbed image).
        :type dataset_descriptor: the descriptor of the dataset.
        :type domain_dimensions (arraylike):  the list of the dimensions to select from.
        """
        self.objective_function_evaluator = objective_function_evaluator
        self.dataset_descriptor = dataset_descriptor
        # total number of BO evaluations performed so far
        self.total_iterations = 0
        self.dimension_bo_runner = DimensionBORunner(domain_dimensions=domain_dimensions)
        # history of inputs
        self.inputs_history = None
        # history of output measurements
        self.outputs_history = None

        # the current status of the attack
        self.attack_status = False

        # the successfully found image
        self.successful_attack_image = None

    # initialize GP and BO with a few initial measurements
    def init_bo(self, initial_dimensions, num_initial_observations, initial_history_inputs, initial_history_outputs):
        self.inputs_history = initial_history_inputs

        self.outputs_history = initial_history_outputs

        observations_per_dimension = int(num_initial_observations / len(initial_dimensions))
        for d in initial_dimensions:
            self._run_with_fixed_dimension(next_dimension=d,
                                           iterations=observations_per_dimension,
                                           early_stop=False)

    # select the next dimension using EI
    def select_next_dimension(self, bos_iterations):
        return self.dimension_bo_runner.select_next_dimension(iterations_run=self.total_iterations + bos_iterations)

    def run(self, bos_iterations, total_iterations_max):
        """
        :param total_iterations_max: max number of BO iterations allowed
        :param bos_iterations: (int): int number of BO iterations allowed for each fixed dimension (inner loop of BO)
        """
        #  check if we are already success after initialisation
        if self.attack_status:
            print("Attack succeeded after {} iterations".format(self.total_iterations))
            return

        while self.total_iterations < total_iterations_max:
            # select the next dimension
            next_dimension = self.select_next_dimension(bos_iterations=bos_iterations)
            # the max number of iterations the BO-BOS algorithm can run
            allowed_iterations = min(total_iterations_max - self.total_iterations, bos_iterations)

            self._run_with_fixed_dimension(next_dimension=next_dimension,
                                           iterations=allowed_iterations,
                                           early_stop=True)
            if self.attack_status:
                break

        # check status after running BO
        if self.attack_status:
            print("Attack succeeded after {} iterations".format(self.total_iterations))
        else:
            print("Attack failed after {} iterations".format(self.total_iterations))

    def _run_with_fixed_dimension(self, next_dimension, iterations, early_stop):
        fixed_dim_runner = FixedDimensionRunner(objective_function_evaluator=self.objective_function_evaluator,
                                                dataset_descriptor=self.dataset_descriptor,
                                                dimension_bo_iteration=self.dimension_bo_runner.iterations_run,
                                                dimension=next_dimension)

        fixed_dim_runner.init(initial_history_inputs=self.inputs_history,
                              initial_history_outputs=self.outputs_history)

        # run BO-BOS for this dimension
        new_inputs, new_outputs = fixed_dim_runner.run(iterations=iterations,
                                                       early_stop=early_stop)

        # add the iterations run by BO-BOS to total
        self.total_iterations += fixed_dim_runner.iterations_run

        # if we found a successful attack, stop
        if fixed_dim_runner.attack_status:
            self.attack_status = True
            self.successful_attack_image = fixed_dim_runner.successful_attack_image
            return

        # if we haven't found the successful attack at this dimension

        # update the outer BO loop with the dimension and best found value for it
        self.dimension_bo_runner.update_history_data(dimension=next_dimension,
                                                     iterations_run=self.total_iterations,
                                                     measurement=fixed_dim_runner.best_output)

        # update the inputs and outputs with the new data obtained from BO-BOS
        # todo change this
        self.inputs_history = np.append(self.inputs_history, new_inputs, axis=0)
        self.outputs_history = np.append(self.outputs_history, new_outputs)
