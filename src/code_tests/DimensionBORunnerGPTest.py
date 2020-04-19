from bo.DimensionBORunner import DimensionBORunner
import numpy as np

if __name__ == '__main__':
    num_points = 1000
    X = np.random.uniform(-3., 3., (num_points, 1))
    # Y = np.sin(X)
    Y = np.sin(X) + np.random.randn(num_points, 1) * 0.05

    bo_runner = DimensionBORunner(bos_iterations=40,
                                  total_iterations_max=50,
                                  domain_dimensions=X)
    bo_runner.history_dimensions = X[:5, :]
    bo_runner.history_measurements = Y[:5, :]

    for i in range(10):
        next_dimension = bo_runner.select_next_dimension()
        bo_runner.history_dimensions = np.append(bo_runner.history_dimensions,
                                                   np.atleast_2d(next_dimension),
                                                   axis=0)

        # new_measurememnt = np.sin(next_dimension) + np.random.randn() * 0.05
        new_measurememnt = np.sin(next_dimension)
        bo_runner.history_measurements = np.append(bo_runner.history_measurements,
                                                   new_measurememnt)

    print(np.max(Y))
    print(max(bo_runner.history_measurements))
