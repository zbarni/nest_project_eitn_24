import numpy as np


########################################################################################################
def generate_piecewise_constant_signal(seed, num_steps, step_duration, resolution, scale):
    """
    Generates a piecewise constant input signal with amplitudes drawn from a uniform distribution

    Parameters
    ----------
    seed: int
        seed to generate the input signal
    num_steps: int
        number of steps in the step function
    step_duration: int
        duration of each step in ms
    resolution: float
        resolution of the generated signal
    scale: float
        amplitude scaling parameter
    Returns
    -------
    ndarray
        continuous input signal (between -1 and 1)
    ndarray
        time vector with all the times for which the signal is generated
    ndarray
        times at which signal amplitude shifts
    ndarray
        amplitudes
    """
    rng = np.random.default_rng(seed)
    dist_range = [-1.0, 1.0]
    rand_distr = rng.uniform(low=dist_range[0], high=dist_range[1], size=num_steps)
    rand_distr = rand_distr + abs(min(dist_range))
    inp_times = np.arange(resolution, num_steps * step_duration, step_duration)
    time_vec = np.arange(0, num_steps * step_duration + resolution, resolution)
    signal = np.zeros_like(time_vec)
    for tt in range(len(inp_times)):
        end_idx = int(round(inp_times[tt + 1] / resolution)) if tt + 1 < len(inp_times) else None
        signal[int(round(inp_times[tt] / resolution)) : end_idx] = rand_distr[tt]

    return signal, time_vec, inp_times, rand_distr * scale
