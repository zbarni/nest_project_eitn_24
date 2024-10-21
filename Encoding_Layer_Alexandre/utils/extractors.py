"""
Helper functions for extracting state variables (V_m, filtered spike trains).
"""

import sys
import time
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import nest


def compile_filtered_spikes_state_matrix(extractor, t_start=None, t_stop=None):
    """
    Parameters
    ----------
    extractor: object
        The spike train extractor object containing events data.
    t_start: float, optional
        The start time for extracting filtered spike trains (default is None).
    t_stop: float, optional
        The stop time for extracting filtered spike trains (default is None).

    """
    print("Extracting filtered spike trains and compiling them into a state matrix...")
    t = time.time()
    neuron_ids = np.arange(len(extractor))
    sampled_times = np.unique(extractor[0].events["times"])
    n_neurons = len(neuron_ids)
    n_samples = len(sampled_times)

    state_matrix = np.zeros((n_neurons, int(n_samples)))
    for i, n in tqdm(enumerate(neuron_ids), total=n_neurons, desc="Extracting"):
        state_matrix[i, :] = extractor[i].events["V_m"]

    # sort first by senders and then by times
    # sorted_idxs = np.lexsort((extractor.events["times"], extractor.events["senders"]))
    # state_matrix = extractor.events["V_m"][sorted_idxs].reshape((n_neurons, int(n_samples)))
    print(f"Elapsed time for extraction: {np.round(time.time() - t, decimals=2)} s")
    return state_matrix


def filter_spikes(spike_times, neuron_ids, n_neurons, t_start, t_stop, dt, tau):
    """
    Returns an NxT matrix where each row represents the filtered spiking activity of
    one neuron and the columns represent time...

    Inputs:
            - spike_times - list of spike times
            - neuron_ids - list of spike ids
            - dt - time step
            - tau - kernel time constant
    """

    neurons = np.unique(neuron_ids)
    new_ids = neuron_ids - min(neuron_ids)
    N = round((t_stop - t_start) / dt)
    state_mat = np.zeros((int(n_neurons), int(N)))

    for i, n in enumerate(tqdm(neurons, desc="Filtering SpikeTrains")):
        idx = np.where(neuron_ids == n)[0]
        spk_times = spike_times[idx]
        state_mat[new_ids[idx][0], :] = spikes_to_states(spk_times, t_start, t_stop, dt, tau)

    return state_mat


def spikes_to_states(spike_times, t_start, t_stop, dt, tau):
    """
    Converts a spike train into an analogue variable (liquid state), by convolving it with an exponential function.
    This process is supposed to mimic the integration performed by the postsynaptic membrane upon an incoming spike.

    Inputs:
            spike_times - array of spike times for a single neuron
            dt     - time step
            tau    - decay time constant
    Examples:
    >> spikes_to_states(spk_times, 0.1, 20.)
    """

    n_spikes = len(spike_times)
    state = 0.0
    # t_start = np.min(spike_times)
    # t_stop = np.max(spike_times)
    N = round((t_stop - t_start) / dt)

    states = np.zeros((1, int(N)))[0]

    time_vec = np.round(np.arange(t_start, t_stop, dt), 1)
    decay = np.exp(-dt / tau)

    if n_spikes:
        idx_spk = 0
        spk_t = spike_times[idx_spk]

        for i, t in enumerate(time_vec):
            if np.round(spk_t, 1) == np.round(t, 1):  # and (idx_Spk<nSpk-1):
                state += 1.0
                if idx_spk < n_spikes - 1:
                    idx_spk += 1
                    spk_t = spike_times[idx_spk]
            else:
                state = state * decay
            if i < int(N):
                states[i] = state

    return states


def filter_spikes_parallel(spike_times, neuron_ids, n_neurons, t_start, t_stop, dt, tau, n_processes):
    """
    Returns an NxT matrix where each row represents the filtered spiking activity of
    one neuron and the columns represent time...

    Inputs:
        - spike_times - list of spike times
        - neuron_ids - list of spike ids
        - dt - time step
        - tau - kernel time constant
        - n_processes - number of processes to use for parallel computation
        - show_progess - if True a progress bar is printed
    """
    spk_times_list = order_array_by_ids(spike_times, n_neurons, neuron_ids)
    arg_list = [
        {
            "spike_times": spkt,
            "t_start": t_start,
            "t_stop": t_stop,
            "dt": dt,
            "tau": tau,
        }
        for spkt in spk_times_list
    ]
    with Pool(n_processes) as p:
        state_mat = list(
            tqdm(p.imap(spikes_to_states_from_dict, arg_list), desc="Filtering SpikeTrains", total=n_neurons)
        )

    return np.array(state_mat)


def spikes_to_states_from_dict(args):
    """
    Helper function to use multiprocessing for filtering the spikes
    """
    return spikes_to_states(args["spike_times"], args["t_start"], args["t_stop"], args["dt"], args["tau"])


def order_array_by_ids(array_to_order, n_possible_ids, ids):
    """
    Orders an array (for example spike trains of neurons) by the given ids (of the neurons).
    Needs the number of possible (neuron) ids, because some ids could be missing (neurons may not have
    fired), but they should be in the resulting list as well.

    Parameters
    ----------
    array_to_order: ndarray of floats
        ndarray with spike times
    n_possible_ids: int
        number of possible ids
    ids: ndarray of int
        ids of the objects to which the elements in the array_to_order belong

    Returns
    -------
    list of ndarrays
        list of spike trains (ndarrays) for each neuron

    Examples
    --------
    >>> spike_times = np.array([10.2, 20.1, 30.1])
    >>> ids = np.array([2, 1, 1])
    >>> order_array_by_ids(spike_times, 3, ids)
    [array([20.1, 30.1]), array([10.2]), array([], dtype=float64)]
    """
    spk_times_list = [np.array([]) for _ in range(n_possible_ids)]
    neurons = np.unique(ids)
    new_ids = ids - min(ids)

    for i, n in enumerate(neurons):
        idx = np.where(ids == n)[0]
        spk_times_list[new_ids[idx[0]]] = array_to_order[idx]

    return spk_times_list


def create_extractor_mm(tau, N, sampling_interval, to_memory=True):
    """
    Creates a layer of delta neurons which can be connected to the circuit for computing the exponentially filtered
    spike traces. The expected behavior is identical to the `filter_spikes` function.

    Extend this function if recording to file is needed
    (see also https://nest-simulator.readthedocs.io/en/stable/auto_examples/multimeter_file.html).

    Parameters
    ----------
    tau
    N
    sampling_interval
    to_memory

    Returns
    -------

    """
    params = {
        "V_m": 0.0,
        "E_L": 0.0,
        "C_m": 1.0,
        "tau_m": tau,
        "V_th": sys.float_info.max,
        "V_reset": 0.0,
        "refractory_input": True,
        "V_min": 0.0,
    }
    print("Adding layer of `iaf_psc_delta` neurons for recording the filtered spike trains")
    filtering_neurons = nest.Create("iaf_psc_delta", N, params=params)
    extractor = nest.Create(
        "multimeter",
        # 1,
        N,
        {
            "record_from": ["V_m"],
            "record_to": "memory" if to_memory else "ascii",
            "interval": sampling_interval,
            # "start": recording_start,
            # "offset": recording_interval,
        },
    )
    return filtering_neurons, extractor
