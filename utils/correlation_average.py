def bin_spikes(spikes, bin_width, min_time):
    senders = spikes.events['senders']
    times = spikes.events['times']
    
    # Create a dictionary to hold spike times for each neuron
    spike_times_by_neuron = {}
    
    # Iterate over senders and times
    for sender, time in zip(senders, times):
        if sender not in spike_times_by_neuron:
            spike_times_by_neuron[sender] = []  # Initialize list if neuron not yet present
        spike_times_by_neuron[sender].append(time)
    
    binned_spike_counts = {}
    # Find the global min and max times for the bins
    max_time = max(times)
    
    # Define bins: from min_time to max_time in steps of 5 ms
    bins = np.arange(min_time, max_time + bin_width, bin_width)
    
    # Iterate over each neuron and bin their spike times
    for neuron, spike_times in spike_times_by_neuron.items():
        # Use np.histogram to bin the spike times for the current neuron
        counts, _ = np.histogram(spike_times, bins=bins)
        
        # Store the binned spike counts for each neuron
        binned_spike_counts[neuron] = counts

    # Get the number of neurons and the number of bins
    neuron_ids = list(binned_spike_counts.keys())
    num_neurons = len(neuron_ids)
    num_bins = len(next(iter(binned_spike_counts.values())))  # Get the number of bins from the first neuron's data
    
    # Create a 2D array to hold the binned counts
    heatmap_data = np.zeros((num_neurons, num_bins))
    
    # Fill the heatmap data array
    for i, neuron in enumerate(neuron_ids):
        heatmap_data[i, :] = binned_spike_counts[neuron]

    return binned_spike_counts, heatmap_data

def average_correlation(matrix, num_pairs, seed=None):
    if seed is not None:
        np.random.seed(seed)  # Set the random seed for reproducibility

    # Randomly select unique pairs of rows
    num_rows = matrix.shape[0]
    if num_pairs > (num_rows * (num_rows - 1)) // 2:
        raise ValueError("num_pairs exceeds the number of unique pairs of rows.")
    # Randomly select 500 unique pairs of rows
    num_rows = matrix.shape[0]
    selected_pairs = set()
    while len(selected_pairs) < num_pairs:
        row1 = np.random.randint(0, num_rows)
        row2 = np.random.randint(0, num_rows)
        # Ensure that we don't select the same row for both pairs (no autocorrelation)
        if row1 != row2:
            selected_pairs.add((row1, row2))
    
    # Compute the correlation for each selected pair
    correlations = []
    for row1, row2 in selected_pairs:
        corr, _ = pearsonr(matrix[row1], matrix[row2])
        correlations.append(corr)
    
    # Average the correlations
    average_correlation = np.mean(correlations)
    return average_correlation
