from enums.GPEnum import GPEnum

from gp.gp_models.AdditiveGP import Additive_GPModel_Learn_Decomp
from gp.gp_models.SimpleGP import GPModel


class GaussianProcessFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_gp(gp_type, noise_var, ARD, seed, sparse, normalize_Y, update_freq,
               nsubspaces):

        gp_exact_feval = False
        gp_noise_var = noise_var

        if noise_var < 1e-6:
            gp_exact_feval = True
            gp_noise_var = None

        if gp_type == GPEnum.AdditiveGP:
            print(f'using additive GP with nsubspaces={nsubspaces}')

            gp = Additive_GPModel_Learn_Decomp(noise_var=gp_noise_var,
                                               exact_feval=gp_exact_feval,
                                               ARD=ARD,
                                               sparse=sparse,
                                               seed=seed,
                                               normalize_Y=normalize_Y,
                                               n_subspaces=nsubspaces,
                                               update_freq=update_freq)
        elif gp_type == GPEnum.SimpleGP:
            print("Using simple GP")

            gp = GPModel(noise_var=noise_var,
                         exact_feval=gp_exact_feval,
                         ARD=ARD,
                         sparse=sparse,
                         seed=seed,
                         normalize_Y=normalize_Y,
                         update_freq=update_freq)
        else:
            raise NotImplementedError
        return gp
