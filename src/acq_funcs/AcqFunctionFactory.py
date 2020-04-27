"""
Generate an acquisition function for performing BO
"""


from acq_funcs.acquisitions.EI import EI
from acq_funcs.acquisitions.LCB import LCB
from acq_funcs.acquisitions.AdditiveLCB import AdditiveLCB
from enums.AcquisitionEnum import AcquisitionEnum


class AcquisitionFunctionFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_acq_function(acq_type, gp_model):
        """
        :param gp_model: The GP model for computing the acq. function.
        :type acq_type: The type of acq. function.

        """
        if acq_type == AcquisitionEnum.EI:
            acq_func = EI(gp_model=gp_model)
        elif acq_type == AcquisitionEnum.LCB:
            acq_func = LCB(gp_model=gp_model)
        elif acq_type == AcquisitionEnum.AdditiveLCB:
            acq_func = AdditiveLCB(gp_model)
        else:
            raise NotImplementedError
        return acq_func
