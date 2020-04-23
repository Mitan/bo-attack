"""
A class for running BO-BOS with fixed dimension
"""

import numpy as np

from bo.VariationalAutoEncoderWrapper import VariationalAutoEncoderWrapper
from bo.bos_function import run_BOS


class BOBOSRunner:
    def __init__(self, dimension, inputs_history, outputs_history, dimension_bo_iteration, initial_bos_iterations=8):
        """

        :type dimension_bo_iteration: int. The current iteration of BO over dimensions. Needed to pass to BOS function
        :type initial_bos_iterations: int. Number of initial BO-BOS iterations to run without stopping.
        :type inputs_history: arraylike. The history of inputs obtained by the previous iterations of BO on images (
        with all other dimensions)
         :type outputs_history: arraylike. The history of outputs obtained by the previous iterations of BO on images (
        with all other dimensions)
        :type dimension: int. The dimension of the inputs to perform BO-BOS

        """
        # the reduced dimension of the inputs to perform BO-BOS
        self.dimension = dimension

        self.vae = VariationalAutoEncoderWrapper(self.dimension)
        # train the VAE
        self.vae.train()

        # the current status of the attack
        self.attack_status = False
        # the best found measurement found by the BO procedure
        self.best_regret = None
        # the image found for the successful attack
        self.successful_attack_image = None
        # the history of inputs found by BO

        # reduce the dimension history of the input using VAE
        self.history_inputs = self.vae.encode_range(inputs=inputs_history)

        # the history of outputs found by BO
        self.history_outputs = np.copy(outputs_history)
        # the history of best outputs found so far needed to take the decision by BO-BOS
        self.history_best_outputs = [np.min(self.history_outputs)]

        # the numbers od iterations run by BO
        self.iterations_run = 0

        # the number of initial BO-BOS iterations to run without stopping.
        self.initial_bos_iterations = initial_bos_iterations

        # grid size for running BOS, see the implementation for more details
        self.BOS_GRID_SIZE = 100
        # todo
        # the bound on summary statistic for BOS
        self.Y_BOUNDS = [-1, 1]

        self.dimension_bo_iteration = dimension_bo_iteration

    # run the BO procedure using BO-BOS
    def run(self, iterations):
        """
        :type iterations: int. Max number of itertions to run BO-BOS
        """
        for i in range(iterations):
            # run an iteration of BO to get a new candidate image
            new_output, new_input = self.get_bo_measurement()

            # update the number of iterations run and the history data
            self.iterations_run += 1
            self.update_history_data(new_input=new_input, new_output=new_output)

            # if we found a successful attack, return
            if new_output < 0:
                self.attack_status = True
                # pass the obtained input through the decoder to get the actual image
                self.successful_attack_image = self.vae.decode(new_input)
                break

            # if we haven't found a successful attack, check if we want to run more iterations using BOS
            # but only if i is larger than initial number of iterations
            if i == self.initial_bos_iterations - 1:
                action_regions, grid_St = run_BOS(init_curve=self.history_best_outputs,
                                                  incumbent=self.best_regret,
                                                  training_epochs=iterations,
                                                  bo_iteration=self.dimension_bo_iteration,
                                                  y_bounds=self.Y_BOUNDS,
                                                  grid_size=self.BOS_GRID_SIZE)

            # start using the decision rules obtained from BOS
            if i >= self.initial_bos_iterations - 1:
                state = np.mean(self.history_best_outputs)
                ind_state = np.max(np.nonzero(state > grid_St)[0])
                action_to_take = action_regions[i, ind_state]

                # condition 1: if action_to_take == 2, then the optimal decision is to stop the current training
                if action_to_take == 2:
                    # condition 2: the second criteria used in the BO-BOS algorithm
                    break

        # after this BO is finished, return the found inputs and outputs
        inputs_found = self.history_inputs[-self.iterations_run:, :]
        outputs_found = self.history_outputs[-self.iterations_run:]
        decoded_inputs_found = self.vae.decode_range(inputs_found)
        return decoded_inputs_found, outputs_found

    def get_bo_measurement(self):
        return 0, 0

    def update_history_data(self, new_input, new_output):
        self.history_inputs = np.append(self.history_inputs, np.atleast_2d(new_input), axis=0)
        self.history_outputs = np.append(self.history_outputs, new_output)
        self.best_regret = np.min(self.history_outputs)
        self.history_best_outputs.append(self.best_regret)
