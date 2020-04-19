"""
A class for running BO-BOS with fixed dimension
"""


class BOBOSRunner:
    def __init__(self, dimension):
        """

        :type dimension: int. The dimension of the inputs to perform BO-BOS
        """
        self.dimension = dimension
        # the current status of the attack
        self.attack_status = False
        # the best found measurement found by the BO procedure
        self.best_measurement = None
        # the image found for the successful attack
        self.succesful_attack_image = None
        # the history of inputs found by BO
        self.inputs_history = None
        # the history of outputs found by BO
        self.outputs_history = None

        # the numbers od iterations run by BO
        self.iterations_run = 0

    # run the BO procedure using BO-BOS
    def run(self, bos_iterations):
        # run BO-BOS
        measurement = 0

        attack_status = False
        iterations_run = 0

        X_new = None
        Y_new = None

        succesful_attack_image = None

        return X_new, Y_new, iterations_run, measurement, attack_status, succesful_attack_image



