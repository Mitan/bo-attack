"""
A class for running BO-BOS with fixed dimension
"""

import numpy as np

from bo.VariationalAutoEncoderWrapper import VariationalAutoEncoderWrapper


class BOBOSRunner:
    def __init__(self, dimension, inputs_history, outputs_history, initial_bos_iterations=8):
        """

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
            # todo check

        # after this BO is finished, return the found inputs and outputs
        inputs_found = self.history_inputs[-self.iterations_run:, :]
        outputs_found = self.history_outputs[-self.iterations_run:]
        self.best_regret = np.min(outputs_found)
        decoded_inputs_found = self.vae.decode_range(inputs_found)
        return decoded_inputs_found, outputs_found

    def get_bo_measurement(self):
        return 0, 0

    def update_history_data(self, new_input, new_output):
        self.history_inputs = np.append(self.history_inputs, np.atleast_2d(new_input), axis=0)
        self.history_outputs = np.append(self.history_outputs, new_output)
        self.history_best_outputs.append(np.min(self.history_outputs))
