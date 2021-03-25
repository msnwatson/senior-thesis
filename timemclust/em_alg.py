from copy import deepcopy
from math import exp, log
from scipy import optimize
from abc import ABC, abstractmethod, abstractproperty
import numpy as np

from timemclust.helper import normalize_data

class ComponentModel(ABC):
    
    # params should be of the form np.ndarray
    @abstractproperty
    def params(self):
        pass

    @abstractmethod
    def log_density(self, time_series, curr_time):
        pass


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
        tau_unnormalized = np.random.uniform(0.25, 0.75, size=self.G)
        return tau_unnormalized / sum(tau_unnormalized)

    # data should be a dict of lists of Unix timestamps
    # only max_methods supported by scipy.optimize 
    def __init__(self, data, G, model, T, epsilon=1e-4, tau=[], max_method='L-BFGS-B', min_steps=50):
        self.data, self.T = normalize_data(data, T)
        self.G = G
        self.epsilon = epsilon
        self.tau = tau
        self.max_method = max_method
        self.n = len(data)
        self.z = np.zeros((self.n, G))
        self.min_steps = min_steps

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


        # optimize for model params
        x0 = []
        bounds = []
        for g in range(self.G):
            x0 += list(self.models[g].params)
            bounds += self.models[g].param_bounds

        maximized_params = optimize.minimize(model_param_func_to_min, x0, args=(self.z, self.data, self.G, self.models, self.T),
                                             method=self.max_method, bounds=bounds)

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
        while (self.param_diff > self.epsilon or self.steps_until_convergence < self.min_steps):
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
