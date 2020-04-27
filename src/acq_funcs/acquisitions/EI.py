"""
This code is modified by Dmitrii based on the original code by Robin Ru
"""
from GPyOpt.util.general import get_quantiles


# All acquisition functions are to be maximised


class EI:

    def __init__(self, gp_model, jitter=0.01):
        """
        EI acquisition function

        :param gp_model: BO surrogate model function
        :param jitter: EI jitter to encourage exploration
        """

        self.gp_model = gp_model
        self.jitter = jitter

    def compute_acq(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        """

        m, s = self.gp_model.predict(x)
        fmin = self.gp_model.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

    def compute_acq_with_gradients(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        :return df_acqu: derivative of acqusition function values w.r.t test location
        """

        fmin = self.gp_model.get_fmin()
        m, s, dmdx, dsdx = self.gp_model.predict_with_gradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu
