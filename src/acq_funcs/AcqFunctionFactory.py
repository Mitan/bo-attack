from acq_funcs.AcquisitionOptimizer import AcqOptimizer
from acq_funcs.Acquisitions import LCB_budget_additive, LCB_budget
from enums.AcquisitionEnum import AcquisitionEnum
from enums.GPEnum import GPEnum


class AcquisitionFunctionFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_acq_optimizer(acq_type, gp_type, gp_model, bounds, nsubspaces):
        # Choose the acquisition function for BO
        if acq_type == AcquisitionEnum.LCB:
            if gp_type == GPEnum.AdditiveGP:
                acq_func = LCB_budget_additive(gp_model)
            else:
                acq_func = LCB_budget(gp_model)
        else:
            raise NotImplementedError

        return AcqOptimizer(model=gp_model,
                            acqu_func=acq_func,
                            bounds=bounds,
                            gp_type=gp_type,
                            nsubspace=nsubspaces)
