import numpy as np
from math import exp, log
from timemclust.em_alg import ComponentModel

def exp_kernel(val, alpha, beta):
    return alpha * beta * exp(-beta * val)

class ExpHawkesProcess(ComponentModel):

    params = None
    param_bounds = None
    kernel = None

    def use_init_params(self):
        # TODO: could express this using more semantics
        # first row is alphas, second is betas, last is baseline
        self.params = np.zeros(3)

        self.params[0] = 1
        self.params[1] = 1
        self.params[2] = 0.1

    def use_init_bounds(self):
        self.param_bounds = [(1e-100, 10), (1e-100, 10), (0, 100)]

    # params kernel should be function
    # param_bounds should be a list of pairs
    def __init__(self, params=None, kernel=exp_kernel, param_bounds=None):
        self.params = params
        self.param_bounds
        self.kernel = kernel

        if params == None:
            self.use_init_params()

        if param_bounds == None:
            self.use_init_bounds()

    def hawkes_cond_intensity(self, current_obs, past_obs):
        # this only the correct value to return in this case
        #  because the conditional intensity only appears in the logarithm
        if len(past_obs) == 0:
            return 1

        ret_value = self.params[2]

        for i in past_obs:
            ret_value += self.kernel(current_obs - i,
                                     self.params[0], self.params[1])

        return ret_value

    # to avoid overflow errors the log density is all that's implemented
    def log_density(self, time_series, curr_time):
        alpha = self.params[0]
        beta = self.params[1]
        baseline = self.params[2]

        first_fact = 0
        k = len(time_series)
        for i in range(k):
            first_fact += log(self.hawkes_cond_intensity(
                time_series[i], time_series[:i]))

        cum_sum = 0
        t_0 = time_series[0]
        for t_i in time_series:
            cum_sum += self.kernel(curr_time - t_i, alpha, beta)

        other_fact = -(curr_time - t_0) * baseline - \
            len(time_series) + cum_sum / beta

        return first_fact + other_fact
