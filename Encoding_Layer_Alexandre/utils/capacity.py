"""
Helper functions for computing the memory capacity of the system.
"""

import numpy as np
import time
from sklearn.linear_model import LinearRegression
from pathos.multiprocessing import ProcessPool as Pool


def compute_capacity(x, z):
    """
    Compute capacity to reconstruct z based on linearly combining x

    Parameters
    ----------
    x : np.ndarray
        state matrix(NxT)
    z : np.ndarray
        target output (1xT)

    Returns
    -------
    z_hat : np.ndarray
        Reconstructed (predicted) signal

    """
    # explicit method 1
    # W_out = np.dot(np.linalg.pinv(x.T), z.T)
    # z_hat = np.dot(W_out, x)

    t_start = time.time()
    reg = LinearRegression(n_jobs=-1, fit_intercept=False).fit(x.T, z)
    W_out = reg.coef_
    z_hat = np.dot(W_out, x)
    # print(f"\nElapsed time for capacity computation: {time.time() - t_start}")

    # capacity = 1.0 - (np.mean((z - z_hat) ** 2) / np.var(z))  # TODO - before
    covs = np.cov(z_hat, z)[0, 1] ** 2.0
    vars = np.var(z) * np.var(z_hat)
    capacity = covs / vars

    error = np.mean((z - z_hat) ** 2)
    return z_hat, capacity, error  # , np.linalg.norm(W_out)


def process_iteration(args):
    # TODO check for correctness
    raise NotImplementedError()
    idx, lag, signal, enc_states, e_states = args

    # shift the target signal
    if idx > 0:
        shifted_signal = signal[:-idx]
    else:
        shifted_signal = signal

    # shift the population states
    enc_st = enc_states[:, idx:]
    circ_st = e_states[:, idx:]

    # compute capacity
    enc_estimate, enc_capacity, enc_error = compute_capacity(enc_st, shifted_signal)
    circ_estimate, circ_capacity, circ_error = compute_capacity(circ_st, shifted_signal)

    result = {
        "lag": lag,
        "enc_capacity": enc_capacity,
        "enc_error": enc_error,
        "circ_capacity": circ_capacity,
        "circ_error": circ_error,
    }

    return result


def parallel_capacity_computation(indices, time_lags, signal, enc_states, e_states):
    # TODO check for correctness
    raise NotImplementedError()
    # Create a list of arguments for each process
    args_list = [(idx, lag, signal, enc_states, e_states) for idx, lag in zip(indices, time_lags)]

    # Use a pool of workers to run the function in parallel
    with Pool() as pool:
        results = pool.map(process_iteration, args_list)

    # Collect and print results
    encoder_capacity = []
    circuit_capacity = []
    for result in results:
        print("Lag = {0} ms".format(str(result["lag"])))
        print(
            "Encoding Layer: \n\t- Capacity={0}, MSE={1}".format(str(result["enc_capacity"]), str(result["enc_error"]))
        )
        print(
            "Main Circuit: \n\t- Capacity={0}, MSE={1}".format(str(result["circ_capacity"]), str(result["circ_error"]))
        )
        encoder_capacity.append(result["enc_capacity"])
        circuit_capacity.append(result["circ_capacity"])

    return encoder_capacity, circuit_capacity


def compute_capacity_sequential(
    time_vector,
    signal,
    enc_states,
    ex_states,
    subsampling_factor,
    max_lag=100.0,
    dt=0.1,
):
    """

    Parameters
    ----------
    time_vector
    signal
    enc_states
    ex_states
    subsampling_factor
    max_lag : float
        Maximum lag / delay for which to evaluate memory capacity
    dt

    Returns
    -------

    """
    # Adjust the maximum lag and step lag based on the subsampling factor
    max_lag_subsampled = max_lag / subsampling_factor  # in milliseconds
    step_lag_subsampled = 10.0 / subsampling_factor  # default step is 10 ms, scaled by subsampling factor

    # Create the array of time lags (subsampled) based on max_lag and step_lag
    time_lags = np.arange(0.0, max_lag_subsampled, step_lag_subsampled)

    # Find indices in the time vector corresponding to these time lags
    indices = [np.where(time_vector == lag)[0][0] for lag in time_lags]

    # Compute the initial index for subsampling based on subsampling factor and dt
    initial_index = max(0, int(subsampling_factor * dt))

    # Generate the indices for subsampling the signal
    subsampling_indices = np.arange(initial_index, len(signal) + initial_index, subsampling_factor, dtype=int)

    # Subsample the signal using the generated indices
    subsampled_signal = signal[subsampling_indices]

    # Initialize lists to store capacities
    encoder_capacity = []
    circuit_capacity = []

    for shift, lag in zip(indices, time_lags):
        # shift the target signal
        if shift > 0:
            shifted_signal = subsampled_signal[:-shift]
        else:
            shifted_signal = subsampled_signal

        # shift the population states
        enc_st = enc_states[:, shift:]
        circ_st = ex_states[:, shift:]

        # compute capacity
        enc_estimate, enc_capacity, enc_error = compute_capacity(enc_st, shifted_signal)
        circ_estimate, circ_capacity, circ_error = compute_capacity(circ_st, shifted_signal)

        print("Lag = {0} ms".format(lag * subsampling_factor))
        print("Encoding Layer: \n\t- Capacity={0}, MSE={1}".format(str(enc_capacity), str(enc_error)))
        print("Main Circuit: \n\t- Capacity={0}, MSE={1}".format(str(circ_capacity), str(circ_error)))

        encoder_capacity.append(enc_capacity)
        circuit_capacity.append(circ_capacity)

        # with open("resultsParamAndDur_networkScale_{0}.csv".format(network_scale), "a") as results_file:
        #     results_file.write(
        #         "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n".format(
        #             str(network_scale),
        #             str(u_low),
        #             str(duration),
        #             str(lag * subsampling_factor),
        #             str(enc_capacity),
        #             str(enc_error),
        #             str(circ_capacity),
        #             str(circ_error),
        #             str(trial),
        #         )
        #     )

    return time_lags * subsampling_factor, encoder_capacity, circuit_capacity
