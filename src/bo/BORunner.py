"""
A class for runner the outer BO loop on dimensions
"""

from bo.DimensionBORunner import DimensionBORunner
import numpy as np

# the paper uses 30 initial evaluations
# 900 max iterations
# GP update frequency of 5 iterations
# learning the dimension with frequency 8 * update_for_hypers, i.e. 8 * 5


class BORunner:
    def __init__(self, domain_dimensions):
        """
        :type domain_dimensions (arraylike):  the list of the dimensions to select from
        """
        # total number of BO evaluations performed so far
        # todo check if we need it
        self.total_iterations = 0
        self.dimension_bo_runner = DimensionBORunner(domain_dimensions=domain_dimensions)
        # history of inputs
        self.inputs_history = None
        # history of output measurements
        self.outputs_history = None

        # the current status of the attack
        self.attack_status = False

        # the succesfully found image
        self.successful_attack_image = None

    # initialize GP and BO with a few initial measurements
    def init_bo(self):
        raise NotImplemented

    # select the next dimension using EI
    def select_next_dimension(self):
        return self.dimension_bo_runner.select_next_dimension()

    def run(self, bos_iterations, total_iterations_max):
        """
        :param total_iterations_max: max number of BO iterations allowed
        :param bos_iterations: (int): int number of BO iterations allowed for each fixed dimension (inner loop of BO)
        """
        # todo check if we are already success
        # todo update this using init. or re-write to self.iterations_run
        total_iterations_run = 0
        while total_iterations_run < total_iterations_max:
            # select the next dimension
            next_dimension = self.select_next_dimension()

            # the max number of iterations the BO-BOS algorithm can run
            allowed_iterations = min(total_iterations_max - total_iterations_run, bos_iterations)

            # run BO-BOS for this dimension
            new_inputs, new_outputs, \
            iterations_run, current_iterations_run, \
            best_measurement, attack_status, succesful_attack_image = \
                self.run_bo_for_fixed_dimension(dimension=next_dimension,
                                                bos_iterations=allowed_iterations)

            # add the iterations run by BO-BOS to total
            total_iterations_run += current_iterations_run

            # if we found a successful attack, stop
            if attack_status:
                self.attack_status = True
                self.successful_attack_image = succesful_attack_image
                break

            # if we haven't found the successful attack at this dimension

            # update the outer BO loop with the dimension and best found value for it
            self.dimension_bo_runner.update_history_data(dimension=next_dimension,
                                                         measurement=best_measurement)

            # update the inputs and outputs with the new data obtained from BO-BOS
            self.inputs_history = np.append(self.inputs_history, new_inputs, axis=0)
            self.outputs_history = np.append(self.outputs_history, new_outputs)

        # check status after running BO
        if self.attack_status:
            print("Attack succeeded after {} iterations".format(total_iterations_run))
        else:
            print("Attack failed after {} iterations".format(total_iterations_run))

    # run the BO procedure using BO-BOS
    def run_bo_for_fixed_dimension(self, dimension, bos_iterations):
        # run BO-BOS
        measurement = 0

        attack_status = False
        iterations_run = 0

        X_new = None
        Y_new = None

        succesful_attack_image = None

        return X_new, Y_new, iterations_run, iterations_run, measurement, attack_status, succesful_attack_image
