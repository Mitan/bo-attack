"""
A class for runner the outer BO loop on dimensions
"""

from bo.DimensionBORunner import DimensionBORunner


class OuterBORunner:
    def __init__(self, domain_dimensions, bos_iterations, total_iterations_max):
        """
        :type total_iterations_max (int): max number of BO iterations allowed
        :type bos_iterations (int): int number of BO iterations allowed for each fixed dimension (inner loop of BO)
        :type domain_dimensions (arraylike):  the list of the dimensions to select from
        """
        self.total_iterations_max = total_iterations_max
        # total number of BO evaluations performed so far
        self.total_iterations = 0
        # the number of iterations for each fixed dimension (inner loop of BO)
        self.bos_iterations = bos_iterations

        self.dimension_bo_runner = DimensionBORunner(domain_dimensions=domain_dimensions)

    # initialize GP and BO with a few initial measurements
    def init_bo(self):
        raise NotImplemented

    # select the next dimension using EI
    def select_next_dimension(self):
        return self.dimension_bo_runner.select_next_dimension()

    # run the BO procedure using BO-BOS
    def run_bo_for_fixed_dimension(self, dimension):
        measurement = 0
        self.dimension_bo_runner.update_history_data(dimension=dimension,
                                                     measurement=measurement)

