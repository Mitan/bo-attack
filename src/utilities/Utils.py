from scipy.stats import norm


def ei_acquizition_function(mu, sigma, best_observation):
    Z = (mu - best_observation) / sigma
    expected_improv = (mu - best_observation) * norm.cdf(x=Z, loc=0, scale=1.0) \
                      + sigma * norm.pdf(x=Z, loc=0, scale=1.0)
    return expected_improv
