from scipy.stats import norm
import numpy as np

def ei_acquizition_function(mu, sigma, best_observation):
    Z = (mu - best_observation) / sigma
    expected_improv = (mu - best_observation) * norm.cdf(x=Z, loc=0, scale=1.0) \
                      + sigma * norm.pdf(x=Z, loc=0, scale=1.0)
    return expected_improv

# taken from BAyesOpt code
def subset_select(X_all, Y_all, select_metric='RAND'):
    """
    Select the subset of the observed data for sparse GP
    :param X_all: observed input data
    :param Y_all: observed output data
    :param select_metric: subset selection criterion
    :return X_ob: subset observed input data
    :return Y_ob: subset observed output data
    """

    N_ob = X_all.shape[0]

    if N_ob <= 500:
        X_ob = X_all
        Y_ob = Y_all
    else:
        # selecting subset if the number of observed data exceeds 500
        if N_ob > 500 and N_ob <= 1000:
            subset_size = 500
        else:
            subset_size = 1000

        print(f'use subset={subset_size} of observed data via {select_metric}')
        if 'SUBRAND' in select_metric:
            x_indices_random = np.random.permutation(range(N_ob))
            x_subset_indices = x_indices_random[:subset_size]
        elif 'SUBGREEDY' in select_metric:
            pseudo_prob_nexp = np.exp(-(Y_all - Y_all.min()))
            pseudo_prob = pseudo_prob_nexp / np.sum(pseudo_prob_nexp)
            x_subset_indices = np.random.choice(N_ob, subset_size, p=pseudo_prob.flatten(), replace=False)
        X_ob = X_all[x_subset_indices, :]
        Y_ob = Y_all[x_subset_indices, :]

    return X_ob, Y_ob

# taken from BAyesOpt code
def subset_select_for_learning(X_all, Y_all, select_metric='ADDRAND'):
    """
    Select the subset of the observed data for sparse GP used only in the phase of learning dr or decomposition
    :param X_all: observed input data
    :param Y_all: observed output data
    :param select_metric: subset selection criterion
    :return X_ob: subset observed input data
    :return Y_ob: subset observed output data
    """
    N_ob = X_all.shape[0]
    subset_size = 200
    pseudo_prob_nexp = np.exp(-(Y_all - Y_all.min()))
    pseudo_prob = pseudo_prob_nexp / np.sum(pseudo_prob_nexp)
    x_subset_indices = np.random.choice(N_ob, subset_size, p=pseudo_prob.flatten(), replace=False)
    X_ob = X_all[x_subset_indices, :]
    Y_ob = Y_all[x_subset_indices, :]
    return X_ob, Y_ob