from tick import hawkes
import numpy as np
from timemclust.em_alg import EMClusterer
from timemclust.exponential_hawkes import ExpHawkesProcess

def two_cluster_sim_data_exp(n_obs, time_series_len, p, a1, a2, b1, b2, mu1, mu2, run_time=100):

    # Define kernels
    h1_kernel = hawkes.HawkesKernelExp(a1, a2)
    h2_kernel = hawkes.HawkesKernelExp(b1, b2)

    sim_data = {}
    labels = []

    coin_flips = np.random.binomial(1, p, n_obs)
    i = 0
    
    for coin_flip in coin_flips:
        if coin_flip == 0:
            h1_simulator = hawkes.SimuHawkes(end_time=run_time, verbose=False, kernels=[[h1_kernel]],
                                             baseline=[mu1], seed=i)

            h1_simulator.simulate()
            lst = h1_simulator.timestamps[0].tolist()[-time_series_len:]
            if (len(lst) > 0):
                sim_data[i] = lst
                labels.append(1)
        else:

            h2_simulator = hawkes.SimuHawkes(end_time=run_time, verbose=False,
                                             seed=i, kernels=[[h2_kernel]],
                                             baseline=[mu2])
            h2_simulator.simulate()

            lst = h2_simulator.timestamps[0].tolist()[-time_series_len:]
            if (len(lst) > 0):
                sim_data[i] = lst
                labels.append(2)

        i += 1

    return sim_data, labels


def perform_tau_diff_exp(taus, sim_data, num_steps=100):
    tau_lst = []

    hawkes_exp_process = ExpHawkesProcess()
    em_clusterer = EMClusterer(
        sim_data, 2, 0, hawkes_exp_process, 100, tau=taus)
    tau_lst.append(em_clusterer.tau)

    for _ in range(num_steps):
        em_clusterer.estep()
        em_clusterer.mstep()
        tau_lst.append(em_clusterer.tau)

    tau_diff = []
    for taus_arr in tau_lst:
        taus_pair = taus_arr.tolist()
        tau_diff.append(abs(taus_pair[0] - taus_pair[1]))

    return tau_diff
