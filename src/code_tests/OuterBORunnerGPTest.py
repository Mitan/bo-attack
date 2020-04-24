from bo.AttackRunner import AttackRunner
import numpy as np

if __name__ == '__main__':
    num_points = 1000
    X = np.random.uniform(-3., 3., (num_points, 1))
    # Y = np.sin(X)
    Y = np.sin(X) + np.random.randn(num_points, 1) * 0.05

    bo_runner = AttackRunner(domain_dimensions=X)
    init_points = 5
    init_history = np.hstack([X[:init_points, :], np.zeros((init_points, 1))])
    bo_runner.dimension_bo_runner.history_dimensions_iterations = init_history

    bo_runner.dimension_bo_runner.history_measurements = Y[:init_points, :]

    dummy_bos_iterations = 5
    for i in range(10):
        next_dimension = bo_runner.select_next_dimension(bos_iterations=dummy_bos_iterations)[0]
        bo_runner.dimension_bo_runner.history_dimensions_iterations = \
            np.append(bo_runner.dimension_bo_runner.history_dimensions_iterations,
                      np.atleast_2d([next_dimension, i * dummy_bos_iterations]),
                      axis=0)

        new_measurememnt = np.sin(next_dimension) + np.random.randn() * 0.05
        # new_measurememnt = np.sin(next_dimension)
        bo_runner.dimension_bo_runner.history_measurements = np.append(
            bo_runner.dimension_bo_runner.history_measurements,
            new_measurememnt)

    print(np.max(Y))
    print(max(bo_runner.dimension_bo_runner.history_measurements))

    # print(bo_runner.dimension_bo_runner.history_dimensions_iterations)