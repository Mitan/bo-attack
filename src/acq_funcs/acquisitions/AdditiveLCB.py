"""
This code is modified by Dmitrii based on the original code by Robin Ru
"""
# All acquisition functions are to be maximised


class AdditiveLCB:

    def __init__(self, gp_model, beta=3):
        """
        Cost-aware LCB acquisition function with additive GP surrogate which encourages low perturbation costs

        :param gp_model: BO surrogate model function
        :param beta: LCB exploration and exploitation trade-off parameter
        """

        self.gp_model = gp_model
        self.beta = beta

    def compute_acq(self, x, subspace_id):
        """
        :param x: test location
        :param subspace_id: select a specific subspace of active dimensions
        :return f_acqu: acqusition function value
        """

        m, s = self.gp_model.predictSub(x, subspace_id=subspace_id)
        f_acqu = - (m - self.beta * s)

        return f_acqu

    def compute_acq_with_gradients(self, x, subspace_id):
        """
        :param x: test location
        :param subspace_id: select a specific subspace of active dimensions
        :return f_acqu: acqusition function value
        :return df_acqu: derivative of acqusition function values w.r.t test location
        """

        m, s, dmdx, dsdx = self.gp_model.predictSub_withGradients(x, subspace_id=subspace_id)
        f_acqu = -m + self.beta * s
        df_acqu = -dmdx + self.beta * dsdx

        return f_acqu, df_acqu
