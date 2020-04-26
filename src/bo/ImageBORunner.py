import numpy as np


# main generic class for running BO
from acq_funcs.AcquisitionOptimizer import AcqOptimizer
from gp.GPFactory import GaussianProcessFactory


class ImageBORunner:
    def __init__(self, dataset_descriptor, initial_history_inputs, initial_history_outputs):
        """
        :type dataset_descriptor: the descriptor of the dataset
        :type initial_history_inputs: array. 2D numpy array of initial inputs
        :type initial_history_outputs: array. 2D numpy array of initial outputs
        """
        self.dataset_descriptor = dataset_descriptor

        assert initial_history_inputs.ndim == 2, "initial_history_inputs has to be 2D array"
        self.history_inputs = initial_history_inputs

        assert initial_history_outputs.ndim == 2, "initial_history_outputs has to be 2D array"
        self.history_outputs = initial_history_outputs

        # the bound for searching the new inputs
        input_dimension = self.history_inputs.shape[1]
        self.x_bounds = np.vstack([[-1, 1]] * input_dimension * dataset_descriptor.nchannel)

        # the GP model for performing BO
        self.gp_model = GaussianProcessFactory().get_gp(gp_type=dataset_descriptor.gp_type,
                                                        noise_var=1.0e-10,
                                                        ARD=False,
                                                        seed=1,
                                                        sparse=None,
                                                        normalize_Y=True,
                                                        update_freq = dataset_descriptor.gp_update_freq,
                                                        nsubspaces=dataset_descriptor.num_subspace)

        # the optimizer for an Acquisition Function
        self.acq_optimizer = AcqOptimizer(acq_type=dataset_descriptor.acq_type,
                                          gp_model=self.gp_model,
                                          bounds=self.x_bounds,
                                          nsubspace=dataset_descriptor.num_subspace)

    def get_next_input(self):
        return self.acq_optimizer.get_next(X=self.history_inputs)

    def update_history_data(self, new_input, new_output):
        # todo update the GP model

        self.history_inputs = np.append(self.history_inputs, np.atleast_2d(new_input), axis=0)
        self.history_outputs = np.append(self.history_outputs, new_output)

    # get the BO results found during the last iterations
    def get_results(self, num_iterations):
        inputs_found = self.history_inputs[-num_iterations:, :]
        outputs_found = self.history_outputs[-num_iterations:]
        return inputs_found, outputs_found
