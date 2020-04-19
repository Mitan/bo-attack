"""
A class for runner the outer BO loop on dimensions
"""


class DimensionBORunner:
    def __init__(self, domain_dimensions, bos_iterations, total_iterations_max):
        """
        :type total_iterations_max: max number of BO iterations allowed
        :type bos_iterations: int number of BO iterations allowed for each fixed dimension (inner loop of BO)
        :type domain_dimensions: arraylike the list of the dimensions to select from
        """
        self.total_iterations_max = total_iterations_max
        # total number of BO evaluations performed so far
        self.total_iterations = 0
        # the number of iterations for each fixed dimension (inner loop of BO)
        self.bos_iterations = bos_iterations
        self.domain_dimensions = domain_dimensions
        # list of selected dimensions
        self.history_dimensions = []
        # list of the measurements of the corresponding dimensions
        self.history_measurements = []
        # the gp for BO on dimensions
        self.dimension_gp = None

    # initialize GP and BO with a few initial measurements
    def init_bo(self):
        raise NotImplemented

    # select the next dimension using EI
    def select_next_dimension(self):
        return 0

    # run the BO procedure using BO-BOS
    def run_bo_for_fixed_dimension(self, dimension):
        raise NotImplemented
