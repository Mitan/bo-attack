# @author: Robin Ru (robin@robots.ox.ac.uk)

import argparse
import os
import pickle

import numpy as np

from dataset_descriptors.DatasetDescriptionFactory import DatasetDescriptionFactory
from enums.AcquisitionEnum import AcquisitionEnum
from enums.DatasetEnum import DatasetEnum
from enums.GPEnum import GPEnum
from src.bayesopt import Bayes_opt
from src.objective_func.objective_functions_tf import CNN
from src.utilities.upsampler import upsample_projection
from src.utilities.utilities import load_or_generate_init_data
from vae_models.vae_mnist_keras import MnistVariationalAutoEncoder


def BayesOpt_attack(dataset_type, model_type, acq_type, low_dim, sparse, seed,
                    img_offset, n_init=50, num_iter=40, ntargets=9, target_label=0, dim_reduction='BILI',
                    obj_metric=1, update_freq=10, nsubspaces=1):
    # Specify code directory
    directory = './'

    dataset_descriptor = DatasetDescriptionFactory().get_dataset_descriptor(dataset_type)

    high_dim = dataset_descriptor.dim
    nchannel = dataset_descriptor.channels
    epsilon = dataset_descriptor.epsilon

    x_bounds = np.vstack([[-1, 1]] * low_dim * nchannel)

    # Specify the experiment results saving directory

    results_data_folder = f'../tests/exp_results/{dataset_descriptor.name_string}_tf_{model_type}_ob{obj_metric}_' \
                          f'_freq{update_freq}_ld{low_dim}_{dim_reduction}/'
    if not os.path.exists(results_data_folder):
        os.makedirs(results_data_folder)

    # Define the model and the original images to be attacked
    cnn = CNN(dataset_name=dataset_type, img_offset=img_offset, epsilon=epsilon,
              dim_reduction=dim_reduction, low_dim=low_dim, high_dim=high_dim,
              obj_metric=obj_metric, results_folder=results_data_folder,
              directory=directory)

    # get and train VAE for dimensionality reduction
    vae = MnistVariationalAutoEncoder(latent_dim=low_dim)
    vae.train(epochs=10)

    # For each image, define the target class
    if ntargets > 1:
        target_list = list(range(ntargets))
    else:
        target_list = [target_label]

    # Start attack each target in sequence
    for tg in target_list:
        cnn.get_data_sample(tg)
        input_label = cnn.input_label
        img_id = cnn.orig_img_id
        target_label = cnn.target_label[0]
        print(f'id={img_offset}, origin={input_label}, target={target_label}, eps={epsilon}, dr={low_dim}')

        # f = lambda x: cnn.np_upsample_evaluate(x)
        f = lambda x: cnn.np_decode_evaluate(x, vae.decoder)

        # Define the name of results file and failure fail(for debug or resume)
        results_file_name = os.path.join(results_data_folder,
                                         f'{model_type}{acq_type}_{dim_reduction}_\
                                         d{low_dim}_i{input_label}_t{target_label}_id{img_id}')
        failed_file_name = os.path.join(results_data_folder,
                                        f'failed_{model_type}{acq_type}_{dim_reduction}_\
                                        d{low_dim}_i{input_label}_t{target_label}_id{img_id}')

        X_opt_all_slices = []
        Y_opt_all_slices = []
        X_query_all_slices = []
        Y_query_all_slices = []
        X_reduced_opt_all_slices = []
        X_reduced_query_all_slices = []

        seed_list = [seed]  # can be modified to do BO over multiple seeds
        for seed in seed_list:
            # Specify the random seed
            np.random.seed(seed)

            x_init, y_init = load_or_generate_init_data(results_file_name=results_file_name,
                                                        f=f,
                                                        n_init=n_init,
                                                        x_bounds=x_bounds)
            print('shapes {} {}'.format(x_init.shape, y_init.shape))

            # Initialise BO
            bayes_opt = Bayes_opt(cnn=cnn, func=f, bounds=x_bounds, saving_path=failed_file_name)
            bayes_opt.initialise(X_init=x_init, Y_init=y_init, gp_type=model_type, acq_type=acq_type,
                                 sparse=sparse, nsubspaces=nsubspaces, update_freq=update_freq,
                                 seed=seed)

            # Run BO
            X_query_full, Y_query, X_opt_full, Y_opt, time_record = bayes_opt.run(total_iterations=num_iter)

            # Reduce the memory needed for storing results
            X_query = X_query_full
            X_opt = X_opt_full[-2:]

            # Store the results
            Y_opt_all_slices.append(Y_opt)
            Y_query_all_slices.append(Y_query)
            opt_dr_list = bayes_opt.opt_dr_list

            X_reduced_opt_all_slices.append(X_opt.astype(np.float16))
            X_reduced_query_all_slices.append(X_query.astype(np.float16))

            # Transform data from reduced search space to original high-dimensional input space
            X_h_query = upsample_projection(dim_reduction, X_query, low_dim=low_dim, high_dim=high_dim,
                                                nchannel=nchannel)
            X_query_all_slices.append(X_h_query)
            X_h_opt = upsample_projection(dim_reduction, X_opt, low_dim=low_dim, high_dim=high_dim,
                                              nchannel=nchannel)
            X_opt_all_slices.append(X_h_opt)
            print(f'Y_opt={Y_opt[-1]}, X_opt shape{X_opt.shape}, X_h_opt shape{X_h_opt.shape}, '
                      f'X_query shape{X_query.shape}, X_h_query shape{X_h_query.shape}')

            # Save the results locally
            results = {'X_opt': X_opt_all_slices,
                       'Y_opt': Y_opt_all_slices,
                       'X_query': X_query_all_slices,
                       'Y_query': Y_query_all_slices,
                       'X_reduced_opt': X_reduced_opt_all_slices,
                       'X_reduced_query': X_reduced_query_all_slices,
                       'dr_opt_list': opt_dr_list,
                       'runtime': time_record}
            with open(results_file_name, 'wb') as file:
                pickle.dump(results, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run BayesOpt Experiments")
    parser.add_argument('-f', '--func', help='Objective function(datasets): mnist, cifar10',
                        default=DatasetEnum.MNIST, type=int)
    parser.add_argument('-m', '--model', help='Surrogate model: GP or ADDGPLD or ADDGPFD or GPLDR',
                        default=GPEnum.SimpleGP, type=int)
    parser.add_argument('-acq', '--acq_func', help='Acquisition function type: LCB, EI',
                        default=AcquisitionEnum.LCB, type=int)

    parser.add_argument('-ld', '--low_dim', help='Dimension of reduced subspace.',
                        default=25, type=int)
    parser.add_argument('-init', '--n_init', help='Initial number of observation.',
                        default=30, type=int)
    parser.add_argument('-nitr', '--max_itr', help='Max BO iterations.',
                        default=900, type=int)
    parser.add_argument('-rd', '--reduction', help='Use which dimension reduction technique. '
                                                   'BILI, BICU, NN, CLUSTER, None.',
                        default='BILI', type=str)
    parser.add_argument('-i', '--img_offset', help='Specify the image id.',
                        default=0, type=int)
    parser.add_argument('-ntg', '--ntargets', help='Number of other targets to be attacked '
                                                   'Set to 9 for MNIST and CIFAR10; to 1 for ImageNet',
                        default=9, type=int)
    parser.add_argument('-tg', '--target_label', help='Target label.',
                        default=0, type=int)
    parser.add_argument('-sp', '--sparse', help='Sparse GP method: subset selection (SUBRAND, SUBGREEDY), '
                                                'subset selection for decomposition/low-dim learning only (ADDSUBRAND), '
                                                'subset selection for both (SUBRANDADD)',
                        default='None', type=str)

    parser.add_argument('-ob', '--obj_metric', help='Metric used to compute objective function.',
                        default=2, type=int)
    parser.add_argument('-freq', '--update_freq', help='Frequency to update the surrogate hyperparameters.',
                        default=5, type=int)
    parser.add_argument('-nsub', '--nsubspace',
                        help='Number of subspaces to be decomposed into only applicable for ADDGP: '
                             'we set to 12 for CIFAR10 or MNIST and 27 for ImageNet',
                        default=12, type=int)
    parser.add_argument('-se', '--seed', help='Random seed', default=1, type=int)

    args = parser.parse_args()
    print(f"Got arguments: \n{args}")
    dataset_type = args.func
    model = args.model
    acq_func = args.acq_func
    n_itrs = args.max_itr
    ntargets = args.ntargets
    target_label = args.target_label
    rd = args.reduction
    low_dim = args.low_dim
    img_offset = args.img_offset
    n_init = args.n_init
    obj_metric = args.obj_metric
    update_freq = args.update_freq
    nsubspace = args.nsubspace
    sparse = args.sparse
    seed = args.seed

    BayesOpt_attack(dataset_type=dataset_type, model_type=model, acq_type=acq_func,
                    low_dim=low_dim, img_offset=img_offset, n_init=n_init, nsubspaces=nsubspace,
                    num_iter=n_itrs, ntargets=ntargets, target_label=target_label, dim_reduction=rd, seed=seed,
                    obj_metric=obj_metric, update_freq=update_freq, sparse=sparse)
