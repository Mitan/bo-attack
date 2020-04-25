import numpy as np


# main generic class for running BO
class ImageBORunner:
    def __init__(self, initial_history_inputs, initial_history_outputs):
        self.history_inputs = initial_history_inputs
        self.history_outputs = initial_history_outputs

    def get_next_input(self):
        return 0

    def update_history_data(self, new_input, new_output):
        self.history_inputs = np.append(self.history_inputs, np.atleast_2d(new_input), axis=0)
        self.history_outputs = np.append(self.history_outputs, new_output)

    # get the BO results found during the last iterations
    def get_results(self, num_iterations):
        inputs_found = self.history_inputs[-num_iterations:, :]
        outputs_found = self.history_outputs[-num_iterations:]
        return inputs_found, outputs_found
