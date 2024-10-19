from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import itertools

class BrunelNetwork:
    def __init__(self, g, eta, J, neuron_params, NE, NI,CE,CI,NrE, NrI, rec_start, rec_stop):
        self.num_ex = NE  # number of excitatory neurons
        self.num_in = NI  # number of inhibitory neurons
        self.c_ex = CE  # number of excitatory connections
        self.c_in = CI  # number of inhibitory connections
        self.J_ex = J  # excitatory weight
        self.J_in = -g*J  # inhibitory weight
        self.n_rec_ex = NrE # number of recorded excitatory neurons, both excitatory and inhibitory
        self.n_rec_in = NrI # number of recorded excitatory neurons, both excitatory and inhibitory
        self.rec_start = rec_start
        self.rec_stop = rec_stop
        self.neuron_params = neuron_params  # neuron params
        self.ext_rate = (self.neuron_params['V_th'] 
                         / (J * self.c_ex * self.neuron_params['tau_m'])
                         * eta * 1000. * self.c_ex)
        
    def create(self):
        # Create the network
        
        # First create the neurons
        self.neurons_ex = nest.Create('iaf_psc_delta', self.num_ex, params=self.neuron_params)
        self.neurons_in = nest.Create('iaf_psc_delta', self.num_in, params=self.neuron_params)
        
        self.neurons = self.neurons_ex + self.neurons_in
        
        # Then create the external spike generator
        self.poisson_noise = nest.Create('poisson_generator', params={'rate': self.ext_rate})
        
        # Then create spike detectors
        self.detector_ex = nest.Create('spike_recorder',
                                       self.n_rec_ex,
                                       params={'start' : self.rec_start, 'stop': self.rec_stop})
        self.detector_in = nest.Create('spike_recorder',
                                       self.n_rec_in,
                                       params={'start': self.rec_start, 'stop': self.rec_stop})
        
        
        # Next we connect the neurons
        nest.Connect(self.neurons_ex, self.neurons_ex,
                     conn_spec = {'rule': 'fixed_indegree',
                                 'indegree': self.c_ex},
                     syn_spec = {'synapse_model': 'static_synapse',
                                 'delay': 1.5,
                                 'weight': self.J_ex})
        nest.Connect(self.neurons_ex, self.neurons_in,
                     conn_spec = {'rule': 'fixed_indegree',
                                 'indegree': self.c_ex},
                     syn_spec = {'synapse_model': 'static_synapse',
                                 'delay': 1.5,
                                 'weight': self.J_ex})
        
        
        nest.Connect(self.neurons_in, self.neurons,
                     conn_spec = {'rule': 'fixed_indegree',
                                 'indegree': self.c_in},
                     syn_spec = {'synapse_model': 'static_synapse',
                                 'delay': 1.5,
                                 'weight': self.J_in})
        
        # Then we connect the external drive to the neurons
        nest.Connect(self.poisson_noise, self.neurons,
                     syn_spec = {'synapse_model': 'static_synapse',
                                 'delay': 1.5,
                                 'weight': self.J_ex})
        
        # Then we connect the the neurons to the spike detectors
        nest.Connect(self.neurons_ex[:self.n_rec_ex], self.detector_ex, 'one_to_one')
        nest.Connect(self.neurons_in[:self.n_rec_in], self.detector_in, 'one_to_one')
        
    def simulate(self, t_sim):
        # Simulate the network with specified 
        nest.Simulate(t_sim)
        
    def get_data(self):
        # get spikes from recorders
        spikes_ex = []
        spikes_in = []
        
        for i in range(self.n_rec_ex):
            spikes_ex.append(
                list(np.sort(nest.GetStatus(self.detector_ex)[i]['events']['times'])))
        for i in range(self.n_rec_in):
            spikes_in.append(
                list(np.sort(nest.GetStatus(self.detector_in)[i]['events']['times'])))
            
        return spikes_ex, spikes_in