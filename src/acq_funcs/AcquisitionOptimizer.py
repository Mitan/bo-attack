"""
Modified by Dmitrii from the original code by Robin Ru
"""
from acq_funcs.AcqFunctionFactory import AcquisitionFunctionFactory
from enums.GPEnum import GPEnum
import numpy as np
from scipy.optimize import fmin_l_bfgs_b


class AcqOptimizer(object):

    def __init__(self, acq_type, bounds, gp_model, nsubspace=1):
        """
        Optimiser for the acquisition functions to recommend the next (batch) locations for evaluation

        :param gp_model: The GP model for computing the acq. function.
        :type acq_type: The type of acq. function.
        :param bounds: input space bounds
        :param nsubspace: number of subspaces needs to be specified for ADDGP-BO but equals 1 for other BO attacks
        """
        self.gp_model = gp_model

        self.bounds = bounds
        self.nsubspace = nsubspace
        self.acq_func = AcquisitionFunctionFactory().get_acq_function(acq_type=acq_type,
                                                                      gp_model=self.gp_model)

    def get_next(self, X):
        """
        :param X: observed input data
        :return new_x: the input recommended by BO to be evaluated next
        :return acq_value: acqusitioin function value of the input recommended
        """
        gp_type = self.gp_model.gp_type
        if gp_type == GPEnum.SimpleGP:
            new_x, acq_value = self._optimise_acqu_func(acqu_func=self.acq_func,
                                                        bounds=self.bounds,
                                                        X_ob=X)

        elif gp_type == GPEnum.AdditiveGP:
            new_x, acq_value = self._optimise_acqu_func_additive(acqu_func=self.acq_func,
                                                                 bounds=self.bounds,
                                                                 X_ob=X,
                                                                 nsubspace=self.nsubspace)
        else:
            raise NotImplementedError

        return new_x, acq_value

    @staticmethod
    # this part is taken from the code of Robin Ru
    def _optimise_acqu_func(acqu_func, bounds, X_ob, func_gradient=True, gridSize=10000, n_start=5):
        """
        Optimise acquisition function built on GP model

        :param acqu_func: acquisition function
        :param bounds: input space bounds
        :param X_ob: observed input data
        :param func_gradient: whether to use the acquisition function gradient in optimisation
        :param gridSize: random grid size
        :param n_start: the top n_start points in the random grid search from which we do gradient-based local optimisation
        :return np.array([opt_location]): global optimum input
        :return f_opt: global optimum
        """

        # Turn the acquisition function to be - acqu_func for minimisation
        target_func = lambda x: - acqu_func._compute_acq(x)

        # Define a new function combingin the acquisition function and its derivative
        def target_func_with_gradient(x):
            acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x)
            return -acqu_f, -dacqu_f

        # Define bounds for the local optimisers
        bounds_opt = list(bounds)

        # Create grid for random search
        d = bounds.shape[0]
        Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                               (gridSize, 1)) * np.random.rand(gridSize, d)
        Xgrid = np.vstack((Xgrid, X_ob))
        results = target_func(Xgrid)

        # Find the top n_start candidates from random grid search to perform local optimisation
        top_candidates_idx = results.flatten().argsort()[
                             :n_start]  # give the smallest n_start values in the ascending order
        random_starts = Xgrid[top_candidates_idx]
        f_min = results[top_candidates_idx[0]]
        opt_location = random_starts[0]

        # Print('done random grid search')
        # Perform multi-start gradient-based optimisation
        for random_start in random_starts:
            if func_gradient:
                x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds_opt,
                                                approx_grad=False, maxiter=5000)
            else:
                x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds_opt,
                                                approx_grad=True, maxiter=5000)
            if f_at_x < f_min:
                f_min = f_at_x
                opt_location = x

        f_opt = - f_min

        return np.array([opt_location]), f_opt

    @staticmethod
    # this part is taken from the code of Robin Ru
    def _optimise_acqu_func_additive(acqu_func, bounds, X_ob, func_gradient=True, gridSize=5000, n_start=1,
                                    nsubspace=12):
        """
        Optimise acquisition function built on ADDGP model

        :param acqu_func: acquisition function
        :param bounds: input space bounds
        :param X_ob: observed input data
        :param func_gradient: whether to use the acquisition function gradient in optimisation
        :param gridSize: random grid size
        :param n_start: the top n_start points in the random grid search from which we do gradient-based local optimisation
        :param nsubspace: number of subspaces in the decomposition
        :return np.array([opt_location]): global optimum input
        :return f_opt: global optimum
        """

        # Create grid for random search
        d = bounds.shape[0]
        Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                               (gridSize, 1)) * np.random.rand(gridSize, d)
        Xgrid = np.vstack((Xgrid, X_ob))
        f_opt_join = []

        # Get the learnt decomposition
        active_dims_list = acqu_func.gp_model.active_dims_list
        opt_location_join_array = np.zeros(d)

        # Optimise the acquisition function in each subspace separately in sequence
        for i in range(nsubspace):
            print(f'start optimising subspace{i}')

            # Define the acquisition function for the subspace and turn it to be - acqu_func for minimisation
            def target_func(x_raw):
                x = np.atleast_2d(x_raw)
                N = x.shape[0]
                if x.shape[1] == d:
                    x_aug = x.copy()
                else:
                    x_aug = np.zeros([N, d])
                    x_aug[:, active_dims_list[i]] = x
                return - acqu_func._compute_acq(x_aug, subspace_id=i)

            # Define a new function combingin the acquisition function and its derivative
            def target_func_with_gradient(x_raw):
                x = np.atleast_2d(x_raw)
                N = x.shape[0]
                if x.shape[1] == d:
                    x_aug = x.copy()
                else:
                    x_aug = np.zeros([N, d])
                    x_aug[:, active_dims_list[i]] = x

                acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x_aug, subspace_id=i)
                return -acqu_f, -dacqu_f

            # Find the top n_start candidates from random grid search to perform local optimisation
            results = target_func(Xgrid)
            top_candidates_idx = results.flatten().argsort()[
                                 :n_start]  # give the smallest n_start values in the ascending order
            random_starts = Xgrid[top_candidates_idx][:, active_dims_list[i]]
            f_min = results[top_candidates_idx[0]]
            opt_location = random_starts[0]

            # Define bounds for the local optimisers for the subspace
            bounds_opt_sub = list(bounds[active_dims_list[i], :])
            for random_start in random_starts:
                if func_gradient:
                    x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds_opt_sub,
                                                    approx_grad=False, maxiter=5000)
                else:
                    x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds_opt_sub,
                                                    approx_grad=True, maxiter=5000)
                if f_at_x < f_min:
                    f_min = f_at_x
                    opt_location = x

            f_opt = -f_min
            opt_location_join_array[active_dims_list[i]] = opt_location
            f_opt_join.append(f_opt)

        f_opt_join_sum = np.sum(f_opt_join)

        return np.atleast_2d(opt_location_join_array), f_opt_join_sum
