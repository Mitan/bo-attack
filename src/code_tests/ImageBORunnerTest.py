from bo.AttackRunner import AttackRunner
import numpy as np

from bo.ImageBORunner import ImageBORunner
from dataset_data.mnist.MnistDescriptor import MNISTDescriptor
from enums.GPEnum import GPEnum

if __name__ == '__main__':
    num_points = 10000
    point_dim = 4
    X = np.random.uniform(-1., 1., (num_points, point_dim))
    # Y = np.sin(X)
    Y = np.sin(X.sum(axis=1, keepdims=True)) + np.random.randn(num_points, 1) * 0.05

    init_points = 5
    init_history_inputs = X[:init_points, :]
    init_history_outputs = Y[:init_points, :]

    dataset_descriptor = MNISTDescriptor()
    dataset_descriptor.gp_type = GPEnum.SimpleGP
    bo_runner = ImageBORunner(initial_history_inputs=init_history_inputs,
                              initial_history_outputs=init_history_outputs,
                              dataset_descriptor=dataset_descriptor)

    dummy_bos_iterations = 5
    for i in range(50):
        next_input = bo_runner.get_next_input()
        new_measurement = np.sin(np.sum(next_input)) + np.random.randn() * 0.05
        bo_runner.update_history_data(new_input=next_input,
                                      new_output=new_measurement)

    print(np.min(Y))
    print(np.min(bo_runner.history_outputs))

    # print(bo_runner.dimension_bo_runner.history_dimensions_iterations)
