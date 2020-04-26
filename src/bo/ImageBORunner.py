import numpy as np


# main generic class for running BO
class ImageBORunner:
    def __init__(self, initial_history_inputs, initial_history_outputs):
        """

        :type initial_history_inputs: array. 2D numpy array of initial inputs
        :type initial_history_outputs: array. 2D numpy array of initial outputs
        """
        assert initial_history_inputs.ndim == 2, "initial_history_inputs has to be 2D array"
        self.history_inputs = initial_history_inputs

        assert initial_history_outputs.ndim == 2, "initial_history_outputs has to be 2D array"
        self.history_outputs = initial_history_outputs

        # the GP model for performing BO
        self.gp_model = None
        # the optimizer for an Acquisition Function
        self.acq_optimizer = None

    def get_next_input(self):
        return self.acq_optimizer.get_next(self.history_inputs, self.history_outputs)

    def update_history_data(self, new_input, new_output):
        # todo update the GP model

        self.history_inputs = np.append(self.history_inputs, np.atleast_2d(new_input), axis=0)
        self.history_outputs = np.append(self.history_outputs, new_output)

    # get the BO results found during the last iterations
    def get_results(self, num_iterations):
        inputs_found = self.history_inputs[-num_iterations:, :]
        outputs_found = self.history_outputs[-num_iterations:]
        return inputs_found, outputs_found
