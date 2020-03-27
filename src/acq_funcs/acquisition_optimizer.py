"""
Modified by Dmitrii from the original code by Robin Ru
"""
from enums.GPEnum import GPEnum
from src.utilities.utilities import optimise_acqu_func, optimise_acqu_func_for_NN, optimise_acqu_func_additive, \
    optimise_acqu_func_mledr


class AcqOptimizer(object):

    def __init__(self, model, acqu_func, bounds, model_name=GPEnum.SimpleGP, nsubspace=1):
        """
        Optimise the acquisition functions to recommend the next (batch) locations for evaluation

        :param model: BO surrogate model function
        :param acqu_func: BO acquisition function
        :param bounds: input space bounds
        :param model_name: the name of the BO surrogate model
        :param nsubspace: number of subspaces needs to be specified for ADDGP-BO but equals 1 for other BO attacks
        """
        self.model = model
        self.acqu_func = acqu_func
        self.bounds = bounds
        self.model_name = model_name
        self.nsubspace = nsubspace

    def get_next(self, X):
        """
        :param X: observed input data
        :return new_x: the input recommended by BO to be evaluated next
        :return acq_value: acqusitioin function value of the input recommended
        """

        if self.model_name == GPEnum.SimpleGP:
            new_x, acq_value = optimise_acqu_func(acqu_func=self.acqu_func, bounds=self.bounds, X_ob=X)
        elif self.model_name == GPEnum.LearnDimGP:
            new_x, acq_value = optimise_acqu_func_mledr(acqu_func=self.acqu_func, bounds=self.bounds,
                                                        X_ob=X)
        elif self.model_name == GPEnum.AdditiveGP:
            new_x, acq_value = optimise_acqu_func_additive(acqu_func=self.acqu_func, bounds=self.bounds,
                                                           X_ob=X, nsubspace=self.nsubspace)
        else:
            new_x, acq_value = optimise_acqu_func_for_NN(acqu_func=self.acqu_func, bounds=self.bounds,
                                                         X_ob=X)

        return new_x, acq_value
