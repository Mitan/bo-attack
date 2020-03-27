#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:45:16 2017

@author: robin
based on GPyOpt
"""

from GPyOpt.util.general import get_quantiles


# All acquisition functions are to be maximised

class EI(object):

    def __init__(self, model, jitter=0.01):
        """
        EI acquisition function

        :param model: BO surrogate model function
        :param jitter: EI jitter to encourage exploration
        """

        self.model = model
        self.jitter = jitter

    def _compute_acq(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        """

        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        :return df_acqu: derivative of acqusition function values w.r.t test location
        """

        fmin = self.model.get_fmin()
        m, s, dmdx, dsdx = self.model.predict_with_gradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu


class LCB(object):

    def __init__(self, model, beta=3):
        """
        LCB acquisition function

        :param model: BO surrogate model function
        :param beta: LCB exploration and exploitation trade-off parameter
        """
        self.model = model
        self.beta = beta

    def _compute_acq(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        """

        m, s = self.model.predict(x)
        f_acqu = - (m - self.beta * s)

        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        :return df_acqu: derivative of acqusition function values w.r.t test location
        """

        m, s, dmdx, dsdx = self.model.predict_with_gradients(x)
        f_acqu = -m + self.beta * s
        df_acqu = -dmdx + self.beta * dsdx
        return f_acqu, df_acqu


class LCB_budget(object):

    def __init__(self, model, beta=3, dis_metric=None):
        """
        Cost-aware LCB acquisition function which encourages low perturbation costs

        :param model: BO surrogate model function
        :param beta: LCB exploration and exploitation trade-off parameter
        :param dis_metric: perturbatino cost metric; if None, the acqusition equals to normal LCB acquisition function
        """
        self.model = model
        self.beta = beta
        self.dis_metric = dis_metric

    def _compute_acq(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        """

        m, s = self.model.predict(x)
        f_acqu = - (m - self.beta * s)

        return f_acqu

    def _compute_acq_withGradients(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        :return df_acqu: derivative of acqusition function values w.r.t test location
        """

        m, s, dmdx, dsdx = self.model.predict_with_gradients(x)
        f_acqu = -m + self.beta * s
        df_acqu = -dmdx + self.beta * dsdx

        return f_acqu, df_acqu


class LCB_budget_additive(object):

    def __init__(self, model, beta=3):
        """
        Cost-aware LCB acquisition function with additive GP surrogate which encourages low perturbation costs

        :param model: BO surrogate model function
        :param beta: LCB exploration and exploitation trade-off parameter
        """

        self.model = model
        self.beta = beta

    def _compute_acq(self, x, subspace_id):
        """
        :param x: test location
        :param subspace_id: select a specific subspace of active dimensions
        :return f_acqu: acqusition function value
        """

        m, s = self.model.predictSub(x, subspace_id=subspace_id)
        f_acqu = - (m - self.beta * s)

        return f_acqu

    def _compute_acq_withGradients(self, x, subspace_id):
        """
        :param x: test location
        :param subspace_id: select a specific subspace of active dimensions
        :return f_acqu: acqusition function value
        :return df_acqu: derivative of acqusition function values w.r.t test location
        """

        m, s, dmdx, dsdx = self.model.predictSub_withGradients(x, subspace_id=subspace_id)
        f_acqu = -m + self.beta * s
        df_acqu = -dmdx + self.beta * dsdx

        return f_acqu, df_acqu
