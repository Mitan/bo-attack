"""
This generic BO class performs minimization of the objective function
"""
import numpy as np

from acq_funcs.AcquisitionOptimizer import AcqOptimizer
from gp.GPFactory import GaussianProcessFactory


class ImageBORunner:
    def __init__(self, dataset_descriptor):
        """
        :type dataset_descriptor: the descriptor of the dataset
        """
        self.dataset_descriptor = dataset_descriptor

        self.history_inputs = None

        self.history_outputs = None

        # the bounds for searching the new inputs
        self.x_bounds = None

        # the GP model for performing BO
        self.gp_model = None

        # the optimizer for an Acquisition Function
        self.acq_optimizer = None
        # the number of iterations run
        self.iterations_run = 0

    def init(self, initial_history_inputs, initial_history_outputs):
        """
            :type initial_history_inputs: array. 2D numpy array of initial inputs
            :type initial_history_outputs: array. 2D numpy array of initial outputs
        """
        assert initial_history_inputs.ndim == 2, "initial_history_inputs has to be 2D array"
        self.history_inputs = initial_history_inputs

        assert initial_history_outputs.ndim == 2, "initial_history_outputs has to be 2D array"
        self.history_outputs = initial_history_outputs
        assert self.history_outputs.shape[0] == self.history_inputs.shape[0]

        # the bounds for searching the new inputs
        input_dimension = self.history_inputs.shape[1]
        print("Init ImageBORunner with dimension {}".format(input_dimension))
        self.x_bounds =  np.vstack([[-1, 1]] * input_dimension * self.dataset_descriptor.channels)

        self.gp_model = GaussianProcessFactory().get_gp(gp_type=self.dataset_descriptor.gp_type,
                                                        noise_var=1.0e-10,
                                                        ARD=False,
                                                        seed=1,
                                                        sparse=None,
                                                        normalize_Y=True,
                                                        update_freq=self.dataset_descriptor.gp_update_freq,
                                                        nsubspaces=self.dataset_descriptor.num_subspace)
        self.gp_model.update_model(X_all=self.history_inputs,
                                   Y_all_raw=self.history_outputs)

        self.acq_optimizer = AcqOptimizer(acq_type=self.dataset_descriptor.acq_type,
                                          gp_model=self.gp_model,
                                          bounds=self.x_bounds,
                                          nsubspace=self.dataset_descriptor.num_subspace)

    def get_next_input(self):
        return self.acq_optimizer.get_next(X=self.history_inputs)

    def update_history_data(self, new_input, new_output):
        self.iterations_run += 1
        self.history_inputs = np.append(self.history_inputs, np.atleast_2d(new_input), axis=0)
        self.history_outputs = np.append(self.history_outputs, np.atleast_2d(new_output), axis=0)
        assert self.history_inputs.shape[0] == self.history_outputs.shape[0]
        self.gp_model.update_model(X_all=self.history_inputs,
                                   Y_all_raw=self.history_outputs,
                                   itr=self.iterations_run)

    # get the BO results found during the last iterations
    def get_results(self, num_iterations):
        inputs_found = self.history_inputs[-num_iterations:, :]
        outputs_found = self.history_outputs[-num_iterations:]
        return inputs_found, outputs_found
