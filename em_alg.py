from copy import deepcopy
from math import exp, log
from scipy import optimize
import numpy as np


def first_event_time(data):
    earliest_per_acc = []
    for _, v in data.items():
        earliest_per_acc.append(min(v))

    return min(earliest_per_acc)


def normalize_data(data, T):
    time_zero = first_event_time(data)
    for k, v in data.items():
        norm_lst = []

        for t in v:
            norm_lst.append(t - time_zero)

        norm_lst.sort()
        data[k] = norm_lst

    return data, T - time_zero


def exp_kernel(val, alpha, beta):
    return alpha * beta * exp(-beta * val)


class ExpHawkesProcess:

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

    # params should be of the form np.ndarray
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


def model_param_func_to_min(model_params, z, data, G, models, T):
    param_dims = len(models[0].params)
    params = model_params.reshape((G, param_dims))

    # store old params just to be extra cautious
    past_params = np.zeros((G, param_dims))
    for g in range(G):
        past_params[g, :] = models[g].params
        models[g].params = params[g, :]

    cum_sum = 0
    i = 0
    for k in data.keys():
        for g in range(G):
            cum_sum += models[g].log_density(data[k], T) * z[i, g]
        i += 1

    # reset params to original values
    for g in range(G):
        models[g].params = past_params[g, :]

    return -cum_sum


class EMClusterer:
    G = 1
    z = None
    tau = None
    models = None
    data = None
    n = None
    T = None
    max_method = ""
    param_diff = 1
    steps_until_convergence = 0

    def gen_initial_tau(self):
        tau_unnormalized = np.random.uniform(size=self.G)
        return tau_unnormalized / sum(tau_unnormalized)

    # data should be a dict of lists of Unix timestamps
    def __init__(self, data, G, epsilon, model, T, tau=[], max_method='L-BFGS-B'):
        self.data, self.T = normalize_data(data, T)
        self.G = G
        self.epsilon = epsilon
        self.tau = tau
        self.max_method = max_method
        self.n = len(data)
        self.z = np.zeros((self.n, G))

        if len(tau) == 0:
            self.tau = self.gen_initial_tau()

        self.models = [model]
        for _ in range(G - 1):
            self.models.append(deepcopy(model))

    def estep(self):
        i = 0
        for k in self.data.keys():

            # compute all the log likelihoods
            log_likes = []
            for j in range(self.G):
                log_likes.append(
                    self.models[j].log_density(self.data[k], self.T))

            log_norm_const = max(log_likes)
            denominator = 0
            for j in range(self.G):
                denominator += self.tau[j] * exp(log_likes[j] - log_norm_const)

            for g in range(self.G):
                numerator = self.tau[g] * exp(log_likes[g] - log_norm_const)
                self.z[i, g] = numerator / denominator

            i += 1

    def mstep(self):
        maximized_tau = np.zeros(self.G)
        n = len(self.data)
        for g in range(self.G):
            cum_sum = 0
            for i in range(n):
                cum_sum += self.z[i, g]

            maximized_tau[g] = (cum_sum / n)

        self.param_diff = np.sum(np.abs(self.tau - maximized_tau))
        self.tau = maximized_tau

        print("new tau:", self.tau)

        # optimize for model params
        x0 = []
        bounds = []
        for g in range(self.G):
            x0 += list(self.models[g].params)
            bounds += self.models[g].param_bounds

        maximized_params = optimize.minimize(model_param_func_to_min, x0, args=(self.z, self.data, self.G, self.models, self.T),
                                             method=self.max_method, bounds=bounds)

        # TODO: do some error handling just not in the case that max iterations exceeded
        argmax_params = maximized_params.x.reshape(
            (self.G, len(self.models[0].params)))

        for g in range(self.G):
            self.models[g].params = argmax_params[g, :]

    def one_step(self):
        self.estep()
        self.mstep()

    def step_until_convergence(self):
        # need to introduce a minimum because sometimes the first
        # few steps are the same
        while (self.param_diff > self.epsilon or self.steps_until_convergence < 200):
            self.one_step()
            self.steps_until_convergence += 1

    def bic(self):
        cum_sum = 0
        for _, v in self.data.items():
            for g in range(self.G):
                cum_sum += self.tau[g] * self.models[g].log_density(v, self.T)

        d = self.G * (1 + len(self.models[0].params))

        return 2 * cum_sum - d * log(self.n)

    def classify(self):
        self.estep()

        labels = []
        for i in range(self.n):
            labels.append(np.argmax(self.z[i, :]))

        return labels
