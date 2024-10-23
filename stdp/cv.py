import numpy as np
import matplotlib.pyplot as plt
import nest

def calculate_cv(spike_times):
    """
    Calculate the coefficient of variation (CV) for a given neuron's spike times.
    
    Parameters:
    spike_times (list of float): List of spike times for a neuron.
    
    Returns:
    float: Coefficient of variation.
    """
    if len(spike_times) < 2:
        return 0.0  # CV is not defined for less than 2 spikes
    
    isi = np.diff(spike_times)  # Inter-spike intervals
    mean_isi = np.mean(isi)
    std_isi = np.std(isi)
    
    cv = std_isi / mean_isi
    return cv

def collect_cvs(spikes):
    """
    Collect the CVs for all neurons.
    
    Parameters:
    spikes (list of list of float): List of spike times for each neuron.
    
    Returns:
    list of float: List of CVs for each neuron.
    """
    cvs = [calculate_cv(spike_times) for spike_times in spikes]
    return cvs

def plot_cv_distribution(cvs, title):
    """
    Plot the distribution of CVs.
    
    Parameters:
    cvs (list of float): List of CVs for each neuron.
    title (str): Title for the plot.
    """
    plt.hist(cvs, bins=30, edgecolor='black')
    plt.xlabel('Coefficient of Variation (CV)')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.show()


# Group spike times by neuron ID
def group_spike_times_by_neuron(spikes):
    senders = spikes['senders']
    times = spikes['times']
    spike_dict = {}
    for sender, time in zip(senders, times):
        if sender not in spike_dict:
            spike_dict[sender] = []
        spike_dict[sender].append(time)
    return spike_dict

