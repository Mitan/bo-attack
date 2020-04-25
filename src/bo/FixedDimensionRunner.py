"""
A class for running BO-BOS with fixed dimension
"""

import numpy as np

from bo.ImageBORunner import ImageBORunner
from bo.BOSFunction import run_BOS
from vae_models.VAEFactory import VAEFactory


class FixedDimensionRunner:
    def __init__(self,
                 dataset_descriptor,
                 dimension,
                 initial_history_inputs,
                 initial_history_outputs,
                 dimension_bo_iteration,
                 initial_bos_iterations=8):
        """

        :type dataset_descriptor: the descriptor of the dataset
        :type dimension_bo_iteration: int. The current iteration of BO over dimensions. Needed to pass to BOS function
        :type initial_bos_iterations: int. Number of initial BO-BOS iterations to run without stopping.
        :type initial_history_inputs: array-like. The history of inputs obtained by the previous iterations of BO
            on images (with all other dimensions)
         :type initial_history_outputs: array-like. The history of outputs obtained by the previous iterations of BO
            on images (with all other dimensions)
        :type dimension: int. The dimension of the inputs to perform BO-BOS

        """
        # the reduced dimension of the inputs to perform BO-BOS
        self.dataset_descriptor = dataset_descriptor
        self.dimension = dimension

        self.vae = VAEFactory.get_vae(dataset_descriptor=self.dataset_descriptor,
                                      latent_dimension=self.dimension)
        # train the VAE
        self.vae.train()

        # the current status of the attack
        self.attack_status = False

        # the image found for the successful attack
        self.successful_attack_image = None

        # reduce the dimension history of the input using VAE
        encoded_initial_history_inputs = self.vae.encode(inputs=initial_history_inputs)

        # the class doing BO on images
        self.image_bo_runner = ImageBORunner(initial_history_inputs=encoded_initial_history_inputs,
                                             initial_history_outputs=initial_history_outputs)

        # the best found measurement found by the BO procedure. Needed for the BO on dimensions
        self.best_output = np.min(initial_history_outputs)

        # the history of best outputs found so far needed to take the decision by BO-BOS
        self.history_best_outputs = [self.best_output]

        # the numbers od iterations run by BO
        self.iterations_run = 0

        # the number of initial BO-BOS iterations to run without stopping.
        self.initial_bos_iterations = initial_bos_iterations

        # grid size for running BOS, see the implementation for more details
        self.BOS_GRID_SIZE = 100
        # todo
        # the bound on summary statistic for BOS
        self.Y_BOUNDS = [-1, 1]

        # The current iteration of BO over dimensions. Needed to pass to BOS function
        self.dimension_bo_iteration = dimension_bo_iteration

    # run the BO procedure using BO-BOS
    def run(self, iterations):
        """
        :type iterations: int. Max number of itertions to run BO-BOS
        """
        for i in range(iterations):
            # run an iteration of BO to get a new candidate image
            new_input = self.image_bo_runner.get_next_input()
            new_output = self.get_bo_measurement(new_input=new_input)

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
                action_regions, grid_st = run_BOS(init_curve=self.history_best_outputs,
                                                  incumbent=self.best_output,
                                                  training_epochs=iterations,
                                                  bo_iteration=self.dimension_bo_iteration,
                                                  y_bounds=self.Y_BOUNDS,
                                                  grid_size=self.BOS_GRID_SIZE)

            # start using the decision rules obtained from BOS
            if i >= self.initial_bos_iterations - 1:
                state = np.mean(self.history_best_outputs)
                ind_state = np.max(np.nonzero(state > grid_st)[0])
                action_to_take = action_regions[i, ind_state]

                # condition 1: if action_to_take == 2, then the optimal decision is to stop the current training
                if action_to_take == 2:
                    # condition 2: the second criteria used in the BO-BOS algorithm
                    break

        # after this BO is finished, return the found inputs and outputs
        decoded_inputs_found, outputs_found = self._get_results()
        return decoded_inputs_found, outputs_found

    def _get_results(self):
        inputs_found, outputs_found = self.image_bo_runner.get_results(num_iterations=self.iterations_run)
        decoded_inputs_found = self.vae.decode(inputs_found)
        return decoded_inputs_found, outputs_found

    def get_bo_measurement(self, new_input):
        return 0

    def update_history_data(self, new_input, new_output):
        self.image_bo_runner.update_history_data(new_input=new_input, new_output=new_output)
        self.best_output = min(self.best_output, new_output)
        self.history_best_outputs.append(self.best_output)
