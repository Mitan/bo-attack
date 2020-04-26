"""
This code is modified by Dmitrii based on the original code by Robin Ru
"""
# All acquisition functions are to be maximised


class LCB:

    def __init__(self, gp_model, beta=3):
        """
        LCB acquisition function

        :param gp_model: BO surrogate model function
        :param beta: LCB exploration and exploitation trade-off parameter
        """
        self.gp_model = gp_model
        self.beta = beta

    def compute_acq(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        """

        m, s = self.gp_model.predict(x)
        f_acqu = - (m - self.beta * s)

        return f_acqu

    def compute_acq_with_gradients(self, x):
        """
        :param x: test location
        :return f_acqu: acqusition function value
        :return df_acqu: derivative of acqusition function values w.r.t test location
        """

        m, s, dmdx, dsdx = self.gp_model.predict_with_gradients(x)
        f_acqu = -m + self.beta * s
        df_acqu = -dmdx + self.beta * dsdx
        return f_acqu, df_acqu
