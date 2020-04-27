"""
A class for runner the outer BO loop on dimensions

NOTE: this BO maximizes the acquisition function.
However, to match the GPy implementation, the target function has to be minimized
Therefore, we change the sign of the output measurements and therefore, maximize the inverted objective function.
This has to be changed later.
#TODO
"""
import math

import GPy
import numpy as np
from utilities.Utils import ei_acquizition_function


class DimensionBORunner:
    def __init__(self, domain_dimensions):
        """
         :type domain_dimensions (list):  the list of the dimensions to select from
        """
        self.domain_dimensions = domain_dimensions
        # list of selected dimensions and corresponding BO iterations run from the beginning
        self.history_dimensions_iterations = None
        # list of the measurements of the corresponding dimensions
        self.history_measurements = None
        # need only to pass bo_iteration to BOS function.
        self.iterations_run = 0

    # initialize GP and BO with a few initial measurements
    # def init(self, initial_dimensions_iterations, initial_measurements):
    #     # todo init GP here and re-learn hypers not every time, but if number of inputs is divisible by certain const
    #     self.history_dimensions_iterations = initial_dimensions_iterations
    #     self.history_measurements = initial_measurements

    # select the next dimension using EI
    def select_next_dimension(self, iterations_run):
        """

        :type iterations_run: int. The number of dimensions to be run after this dimension is explored.
        Is equal to the current number of iterations run + the number of iterations run by BO-BOS
        """
        num_points = self.history_measurements.shape[0]
        assert num_points == self.history_dimensions_iterations.shape[0]

        # get a GP for the next
        # ker = GPy.kern.RBF(input_dim=1)
        # the inputs have dimension 2 - (dimension, number of BO iterations)
        ker = GPy.kern.Matern52(input_dim=2, variance=1.0, lengthscale=1.0)

        m = GPy.models.GPRegression(self.history_dimensions_iterations,
                                    self.history_measurements.reshape(num_points, -1),
                                    kernel=ker)
        m.likelihood.variance.fix(1e-3)
        # m.optimize_restarts(num_restarts = 10, messages=False)
        m.optimize(messages=False)

        best_observation = max(self.history_measurements)

        best_dim = None
        best_val = -float('inf')
        for next_dim in self.domain_dimensions:
            next_dim_it = np.atleast_2d([next_dim, iterations_run])
            mu, var = m.predict(next_dim_it)
            # todo how to write it in a better way?
            std = math.sqrt(var[0, 0])
            mu = mu[0, 0]
            ei = ei_acquizition_function(mu=mu,
                                         sigma=std,
                                         best_observation=best_observation)
            if ei > best_val:
                best_dim = next_dim
                best_val = ei
        # update the number of iterations. Not sure if here is the best place to do it.
        self.iterations_run += 1
        return best_dim

    def update_history_data(self, dimension, iterations_run, measurement):
        if self.history_dimensions_iterations is None:
            self.history_dimensions_iterations = np.atleast_2d([dimension, iterations_run])
            self.history_measurements = np.array(- measurement)
        else:
            self.history_dimensions_iterations = np.append(self.history_dimensions_iterations,
                                                           np.atleast_2d([dimension, iterations_run]), axis=0)
            # TODO note the inverted sign. See the header of this file fore more details
            self.history_measurements = np.append(self.history_measurements, - measurement)
