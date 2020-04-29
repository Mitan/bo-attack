"""
A class for running BO-BOS with fixed dimension
"""

import numpy as np

from src.bo.ImageBORunner import ImageBORunner
from src.bo.BOSFunction import run_BOS
from src.vae_models.VAEFactory import VAEFactory


class FixedDimensionRunner:
    def __init__(self,
                 objective_function_evaluator,
                 dataset_descriptor,
                 dimension,
                 dimension_bo_iteration,
                 initial_bos_iterations=8):
        """

        :type objective_function_evaluator: evaluator of the objective function (the score of the perturbed image).
        :type dataset_descriptor: the descriptor of the dataset
        :type dimension_bo_iteration: int. The current iteration of BO over dimensions. Needed to pass to BOS function
        :type initial_bos_iterations: int. Number of initial BO-BOS iterations to run without stopping.
        :type dimension: int. The dimension of the inputs to perform BO-BOS

        """
        # the evaluator of the objective function
        self.objective_function_evaluator = objective_function_evaluator
        # the dataset descriptor
        self.dataset_descriptor = dataset_descriptor
        # the reduced dimension of the inputs to perform BO-BOS
        self.dimension = dimension

        self.vae = VAEFactory.get_vae(dataset_descriptor=self.dataset_descriptor,
                                      latent_dimension=self.dimension)

        # the current status of the attack
        self.attack_status = False

        # the image found for the successful attack
        self.successful_attack_image = None

        # the class doing BO on images
        self.image_bo_runner = None

        # the min found measurement found by the BO procedure. Needed for the BO on dimensions
        self.best_output = None

        # the history of best outputs found so far needed to take the decision by BO-BOS
        self.history_best_outputs = None

        # the numbers od iterations run by BO
        self.iterations_run = 0

        # the number of initial BO-BOS iterations to run without stopping.
        self.initial_bos_iterations = initial_bos_iterations

        # grid size for running BOS, see the implementation for more details
        self.BOS_GRID_SIZE = 100
        # todo
        # the bound on summary statistic for BOS
        self.Y_BOUNDS = dataset_descriptor.y_bounds_bos

        # The current iteration of BO over dimensions. Needed to pass to BOS function
        self.dimension_bo_iteration = dimension_bo_iteration

    # init
    def init(self, initial_history_inputs, initial_history_outputs):
        """
        :type initial_history_inputs: array-like. The history of inputs obtained by the previous iterations of BO
            on images (with all other dimensions)
        :type initial_history_outputs: array-like. The history of outputs obtained by the previous iterations of BO
            on images (with all other dimensions)
        """
        # self.vae.train(num_epochs=self.dataset_descriptor.vae_num_epochs,
        #                dataset_folder=self.dataset_descriptor.dataset_folder)
        # all VAE are pre-trained and their weights are loaded instead of training
        self.vae.load_weights(load_folder=self.dataset_descriptor.vae_weights_folder,
                              num_epochs_trained=self.dataset_descriptor.vae_num_epochs)

        # reduce the dimension history of the input using VAE
        encoded_initial_history_inputs = self.vae.encode(inputs=initial_history_inputs)

        # the class for doing BO on images
        self.image_bo_runner = ImageBORunner(dataset_descriptor=self.dataset_descriptor)

        self.image_bo_runner.init(initial_history_inputs=encoded_initial_history_inputs,
                                  initial_history_outputs=initial_history_outputs)

        self.best_output = np.min(initial_history_outputs)
        self.history_best_outputs = np.array(self.best_output)

    # run the BO procedure using BO-BOS
    def run(self, iterations, early_stop):
        """
        :type early_stop: bool. flag to indicate whether we want to run BOS or not
         (we don't want to run it during the initialisation)
        :type iterations: int. Max number of itertions to run BO-BOS
        """

        for i in range(iterations):
            # run an iteration of BO to get a new candidate image
            new_input = self.image_bo_runner.get_next_input()
            new_output = self.get_bo_measurement(new_input=new_input)

            # update the number of iterations run and the history data
            self.iterations_run += 1


            # if we found a successful attack, return
            if new_output < 0:
                self.attack_status = True
                # pass the obtained input through the decoder to get the actual image
                self.successful_attack_image = self.vae.decode(new_input)
                break

            # if we haven't found a successful attack, check if we want to run more iterations using BOS
            # but only if i is larger than initial number of iterations and we are using early stopping with BOS
            if early_stop and (i == self.initial_bos_iterations - 1):
                # BOS implementation requires the learning curves to decrease
                action_regions, grid_st = run_BOS(init_curve=self.history_best_outputs,
                                                  incumbent=self.best_output,
                                                  training_epochs=iterations,
                                                  bo_iteration=self.dimension_bo_iteration,
                                                  y_bounds=self.Y_BOUNDS,
                                                  grid_size=self.BOS_GRID_SIZE)

            self.update_history_data(new_input=new_input, new_output=new_output)

            # start using the decision rules obtained from BOS (only if  we are using early stopping with BOS)
            if early_stop and (i >= self.initial_bos_iterations):
                # BOS implementation requires the learning curves to decrease, so invert them
                state = np.mean(self.history_best_outputs)
                ind_state = np.max(np.nonzero(state > grid_st)[0])

                action_to_take = action_regions[i - self.initial_bos_iterations, ind_state]

                # condition 1: if action_to_take == 2, then the optimal decision is to stop the current training
                if action_to_take == 2:
                    print("Stopping BOS for dimension {} after {} iterations".format(self.dimension, i))
                    # condition 2: the second criteria used in the BO-BOS algorithm
                    break

        # after this BO is finished, return the found inputs and outputs
        decoded_inputs_found, outputs_found = self._get_results()
        return decoded_inputs_found, outputs_found

    def _get_results(self):
        inputs_found, outputs_found = self.image_bo_runner.get_results(num_iterations=self.iterations_run)
        decoded_inputs_found = self.vae.decode(inputs_found)
        return decoded_inputs_found, outputs_found

    # evaluate the currently found input perturbation
    def get_bo_measurement(self, new_input):
        decoded_new_input = self.vae.decode(new_input)
        return self.objective_function_evaluator.evaluate(decoded_new_input)

    def update_history_data(self, new_input, new_output):
        # update the image BO runner with the new encoded image and the output
        self.image_bo_runner.update_history_data(new_input=new_input, new_output=new_output)
        # update the best output for dimension BO
        self.best_output = min(self.best_output, new_output)
        # update the history of best outputs for performing BOS
        self.history_best_outputs = np.append(arr=self.history_best_outputs, values=self.best_output)
