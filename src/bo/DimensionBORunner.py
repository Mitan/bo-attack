"""
A class for runner the outer BO loop on dimensions
"""
import math

import GPy
import numpy as np
from utilities.Utils import ei_acquizition_function


class DimensionBORunner:
    def __init__(self, domain_dimensions):
        """
         :type domain_dimensions (arraylike):  the list of the dimensions to select from
        """
        self.domain_dimensions = domain_dimensions

        # list of selected dimensions
        self.history_dimensions = []
        # list of the measurements of the corresponding dimensions
        self.history_measurements = []

    # initialize GP and BO with a few initial measurements
    def init_bo(self):
        raise NotImplemented

    # select the next dimension using EI
    def select_next_dimension(self):
        num_points = self.history_measurements.shape[0]
        assert num_points == self.history_dimensions.shape[0]

        # get a GP for the next
        # ker = GPy.kern.RBF(input_dim=1)
        ker =GPy.kern.Matern52(input_dim=1, variance=1.0, lengthscale=1.0)

        m = GPy.models.GPRegression(self.history_dimensions,
                                    self.history_measurements.reshape(num_points, -1),
                                    kernel=ker)
        m.likelihood.variance.fix(1e-3)
        # m.optimize_restarts(num_restarts = 10, messages=False)
        m.optimize(messages=False)

        best_observation = max(self.history_measurements)

        best_x = None
        best_val = -float('inf')
        for next_x in self.domain_dimensions:
            mu, var = m.predict(np.atleast_2d(next_x))
            # todo how to write it in a better way?
            std = math.sqrt(var[0,0])
            mu = mu[0,0]
            ei = ei_acquizition_function(mu=mu,
                                         sigma=std,
                                         best_observation=best_observation)
            if ei > best_val:
                best_x = next_x
                best_val = ei
        return best_x
