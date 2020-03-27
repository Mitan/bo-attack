from enum.GPEnum import GPEnum
from gp.additive_gp_decomp import Additive_GPModel_Learn_Decomp
from gp.gpdr import GPModelLDR


class GaussianProcessFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_gp(gp_type, noise_var, ARD, seed, high_dim, dim_reduction, sparse, normalize_Y, update_freq,
               nchannel, nsubspaces):
        if gp_type == GPEnum.LearnDimGP:
            if noise_var > 1e-6:
                gp = GPModelLDR(noise_var=noise_var, ARD=ARD, seed=seed, high_dim=high_dim,
                                dim_reduction=dim_reduction, sparse=sparse,
                                normalize_Y=normalize_Y, update_freq=update_freq, nchannel=nchannel)
            else:
                gp = GPModelLDR(exact_feval=True, ARD=ARD, seed=seed, high_dim=high_dim,
                                dim_reduction=dim_reduction, sparse=sparse,
                                normalize_Y=normalize_Y, update_freq=update_freq, nchannel=nchannel)

        elif gp_type == GPEnum.AdditiveGP:
            print(f'nsubspaces={nsubspaces}')
            if noise_var > 1e-6:
                gp = Additive_GPModel_Learn_Decomp(noise_var=noise_var, ARD=ARD,
                                                   sparse=sparse,
                                                   seed=seed, normalize_Y=normalize_Y,
                                                   n_subspaces=nsubspaces, update_freq=update_freq)
            else:
                gp = Additive_GPModel_Learn_Decomp(exact_feval=True, ARD=ARD, sparse=sparse,
                                                   seed=seed, normalize_Y=normalize_Y,
                                                   n_subspaces=nsubspaces, update_freq=update_freq)

        return gp
