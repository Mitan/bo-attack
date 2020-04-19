# @author: Robin Ru (robin@robots.ox.ac.uk)

import pickle
import time

import numpy as np

from acq_funcs.AcqFunctionFactory import AcquisitionFunctionFactory
from enums.AcquisitionEnum import AcquisitionEnum
from enums.GPEnum import GPEnum
from gp.gp_factory import GaussianProcessFactory


class Bayes_opt():
    def __init__(self, cnn, func, bounds, saving_path):
        """
        Bayesian Optimisation algorithm

        :type cnn: the cnn to be attacked. Needed only to get attack status
        :param func: the objective function to be optimised
        :param bounds: the input space bounds
        :param saving_path: saving path for failed BO runs (rarely occurred)
        """

        self.func = func
        self.bounds = bounds
        self.noise_var = 1.0e-10
        self.saving_path = saving_path
        self.gp_factory = GaussianProcessFactory()
        self.acq_factory = AcquisitionFunctionFactory()
        # TODO fix
        self.cnn = cnn

    def initialise(self,
                   X_init=None,
                   Y_init=None,
                   gp_type=GPEnum.SimpleGP,
                   acq_type=AcquisitionEnum.LCB,
                   sparse=None,
                   seed=42,
                   ARD=False,
                   nsubspaces=1,
                   normalize_Y=True,
                   update_freq=10):
        """
        :param X_init: initial observation input data
        :param Y_init: initial observation input data
        :param gp_type: BO surrogate model type
        :param acq_type: BO acquisition function type
        :param batch_size: the number of new query locations in the batch (=1 for sequential BO and > 1 for parallel BO)
        :param sparse: sparse GP options
        :param seed: random seed
        :param ARD: ARD option for GP models
        :param cost_metric: perturbatino cost metric; if None, the acqusition equals to normal LCB acquisition function
        :param nsubspaces: number of subspaces in the decomposition for ADDGP only
        :param normalize_Y: normalise output data
        :param update_freq: frequency of relearning GP hyperparameters
        """
        assert X_init.ndim == 2, "X_init has to be 2D array"
        assert Y_init.ndim == 2, "Y_init has to be 2D array"

        self.X_init = X_init
        self.Y_init = Y_init
        self.X = np.copy(X_init)
        self.Y = np.copy(Y_init)
        self.acq_type = acq_type
        self.gp_type = gp_type
        self.seed = seed
        self.X_dim = self.X.shape[1]

        # Find the minimum observed functional value and its location
        self.arg_opt = np.atleast_2d(self.X[np.argmin(self.Y)])
        self.minY = np.min(self.Y)

        self.gp_model = self.gp_factory.get_gp(gp_type=gp_type,
                                               noise_var=self.noise_var,
                                               ARD=ARD,
                                               seed=seed,
                                               sparse=sparse,
                                               normalize_Y=normalize_Y,
                                               update_freq=update_freq,
                                               nsubspaces=nsubspaces)

        self.acq_optimizer = self.acq_factory.get_acq_optimizer(acq_type=acq_type,
                                                                gp_model=self.gp_model,
                                                                gp_type=gp_type,
                                                                bounds=self.bounds,
                                                                nsubspaces=nsubspaces)

    def run(self, total_iterations):
        """
        :param total_iterations:
        # X_query, Y_query - query points selected by BO;
            # X_opt, Yopt      - guesses of the global optimum/optimiser (= optimum point of GP posterior mean)
        :return X_query: inputs queried by BO;
        :return Y_query: output values at queried locations
        :return X_opt: the guess of the global optimum location
        :return Yopt: the guess of the global optimum value
        :return time_record: BO time array for all iterations
        """

        np.random.seed(self.seed)
        X_query = np.copy(self.X)
        Y_query = np.copy(self.Y)
        X_opt = np.copy(np.atleast_2d(self.arg_opt))
        Y_opt = np.copy(np.atleast_2d(self.minY))
        time_record = np.zeros([total_iterations, 2])
        self.opt_dr_list = []

        # # Upsample the observed data to image dimension in the case of auto-learning of d^r
        # if self.gp_type == GPEnum.LearnDimGP:
        #     x_curr_dim = self.X.shape[1]
        #     if int(x_curr_dim / self.nchannel) < self.high_dim:
        #         self.X = upsample_projection(self.dim_reduction, X_query, low_dim=int(x_curr_dim / self.nchannel),
        #                                      high_dim=self.high_dim, nchannel=self.nchannel)

        # Fit GP model to the observed data
        self.gp_model._update_model(self.X, self.Y, itr=0)

        for k in range(total_iterations):

            # Optimise the acquisition function to get the next query point and evaluate at next query point
            start_time_opt = time.time()
            x_next, _ = self.acq_optimizer.get_next(self.X)

            t_opt_acq = time.time() - start_time_opt
            time_record[k, 0] = t_opt_acq

            # # Upsample the observed data to image dimension in the case of auto-learning of d^r after each iteration
            # if self.gp_type == GPEnum.LearnDimGP:
            #     self.opt_dr_list.append(self.gp_model.opt_dr)
            #     x_curr_dim = x_next.shape[1]
            #     if int(x_curr_dim / self.nchannel) < self.high_dim:
            #         x_next = upsample_projection(self.dim_reduction, x_next,
            #                                            low_dim=int(x_curr_dim / self.nchannel), high_dim=self.high_dim,
            #                                            nchannel=self.nchannel)
            # else:
            self.opt_dr_list.append(np.atleast_2d(0))

            # Evaluate the objective function at the next query point
            y_next = self.func(x_next) + np.random.normal(0, np.sqrt(self.noise_var),
                                                                      (x_next.shape[0], 1))
            # Augment the observed data
            self.X = np.vstack((self.X, x_next))
            self.Y = np.vstack((self.Y, y_next))
            self.minY = np.min(self.Y)

            #  Store the intermediate BO results
            X_query = np.vstack((X_query, np.atleast_2d(x_next)))
            Y_query = np.vstack((Y_query, np.atleast_2d(y_next)))
            X_opt = np.concatenate((X_opt, np.atleast_2d(X_query[np.argmin(Y_query), :])))
            Y_opt = np.concatenate((Y_opt, np.atleast_2d(min(Y_query))))

            print(f'{self.gp_type}{self.acq_type} ||'
                  f'seed:{self.seed},itr:{k}, y_next:{np.min(y_next)}, y_opt:{Y_opt[-1, :]}')

            # Terminate the BO loop if the attack succeeds
            # todo do we really need cnn here
            if min(Y_query) <= 0 or self.cnn.success:
                break

            # Update the surrogate model with new data
            # start_time_update = time.time()
            try:
                self.gp_model._update_model(self.X, self.Y, itr=k)
            except:
                # If the model update fails, terminate the BO loop
                partial_results = {'X_query': self.X.astype(np.float16),
                                   'Y_query': self.Y.astype(np.float16),
                                   'model_kernel': self.gp_model.model.kern}
                failed_file_name = self.saving_path
                with open(failed_file_name, 'wb') as file:
                    pickle.dump(partial_results, file)
                print('This BO target failed')
                assert False
            # t_update_model = time.time() - start_time_update
            # time_record[k, 1] = t_update_model
            # print(f'Time for optimising acquisition function={t_opt_acq}; '
            #       f'Time for updating the model={t_update_model}')

        return X_query, Y_query, X_opt, Y_opt, time_record