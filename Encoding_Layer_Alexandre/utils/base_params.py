# neuron paramters
neuron_params = {
    "C_m": 1.0,  # membrane capacity (pF)
    "E_L": 0.0,  # resting membrane potential (mV)
    "I_e": 0.0,  # external input current (pA)
    "V_m": 0.0,  # membrane potential (mV)
    "V_reset": 10.0,  # reset membrane potential after a spike (mV)
    "V_th": 20.0,  # spike threshold (mV)
    "t_ref": 2.0,  # refractory period (ms)
    "tau_m": 20.0,  # membrane time constant (ms)
}

params = {
    "num_neurons": 1250,  # number of neurons in network
    "rho": 0.2,  # fraction of inhibitory neurons
    "eps": 0.2,  # probability to establish a connections
    "g": 5.0,  # excitation-inhibition balance
    "eta": 0.0,  # relative external rate
    "J": 0.1,  # postsynaptic amplitude in mV
    "delay": 1.5,  # synaptic delay (dendritic)
    "neuron_params": neuron_params,  # single neuron parameters
    "n_rec_ex": 500,  # excitatory neurons to be recorded from
    "n_rec_in": 4,  # inhibitory neurons to be recorded from
    # "rec_start": 0.0,  # start point for recording spike trains
    # "rec_stop": 2000.0,  # end points for recording spike trains
}
