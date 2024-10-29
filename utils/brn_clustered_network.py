""" Camila Losada 2024/10/23 """
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import itertools
from utils.inputs import generate_piecewise_constant_signal

class BrunelClusterNetwork:
    def __init__(self, g, eta,w, w_e,w_i, neuron_params, NE, NI,CE,CI,NrE, NrI, rec_start, rec_stop,input_s,dt,scalef,seed=1):
        self.seed=seed
        self.num_ex = NE  # number of excitatory neurons
        self.num_in = NI  # number of inhibitory neurons
        self.c_ex = CE  # number of excitatory connections
        self.c_in = CI  # number of inhibitory connections
        self.w_e = w_e  # excitatory weight
        self.w_i = w_i  # inhibitory weight
        self.w = w
        self.n_rec_ex = NrE #! Not used: number of recorded excitatory neurons
        self.n_rec_in = NrI #! Not used: number of recorded in inhibitory neurons
        self.rec_start = rec_start
        self.rec_stop = rec_stop
        self.neuron_params = neuron_params  # neuron params
        self.input_s=input_s
        self.dt=dt
        self.scalef=scalef 
        self.ext_rate = (self.neuron_params['V_th'] 
                         / (self.w * self.c_ex * self.neuron_params['tau_m'])
                         * eta * 1000. * self.c_ex)
 
    def create_network(self):
        # Create the network
        # First create the neurons
        self.neurons_ex = nest.Create('iaf_psc_delta', self.num_ex, params=self.neuron_params)
        self.neurons_in = nest.Create('iaf_psc_delta', self.num_in, params=self.neuron_params)
        self.neurons = self.neurons_ex + self.neurons_in
        # Second create spike detectors
        self.recorder_ex = nest.Create('spike_recorder',
                                       params={'start' : self.rec_start, 'stop': self.rec_stop})
        self.recorder_in = nest.Create('spike_recorder',
                                       params={'start': self.rec_start, 'stop': self.rec_stop})

        # 3rd connect the neurons
        # E-E
        for i_row in range(self.num_ex):
            # select non-zero weights
            nonzero = self.w_e != 0
            nonzero_row = nonzero[i_row]
            nest.Connect(self.neurons_ex[i_row], self.neurons_ex[nonzero_row],
                        conn_spec = {'rule': 'all_to_all'},
                        syn_spec = {'synapse_model': 'static_synapse',
                                    'delay': 1.5,
                                    'weight': self.w_e[i_row][nonzero_row].reshape((-1,1))})      
        # E-I
        nest.Connect(self.neurons_ex, self.neurons_in,
                     conn_spec = {'rule': 'fixed_indegree',
                                 'indegree': self.c_ex},
                     syn_spec = {'synapse_model': 'static_synapse',
                                 'delay': 1.5,
                                 'weight': self.w})
        # I-I and I-E (I to all)
        nest.Connect(self.neurons_in, self.neurons,
                     conn_spec = {'rule': 'fixed_indegree',
                                 'indegree': self.c_in},
                     syn_spec = {'synapse_model': 'static_synapse',
                                 'delay': 1.5,
                                 'weight': self.w_i}) 
                           
                                 
        # 4th connect the neurons to the spike recorders
        nest.Connect(self.neurons_ex, self.recorder_ex)
        nest.Connect(self.neurons_in, self.recorder_in)
        
        # 5th connect voltimeter for the decoder
        self.vm = nest.Create('multimeter', 1, {'record_from': ['V_m'], 'interval': 1})
        nest.Connect(self.vm, self.neurons_ex)

    def generate_input(self,sim_time):           
        step_duration = 20.    
        scale = self.ext_rate*self.scalef # input scaling factor [Hz]
        num_steps = int( sim_time/  step_duration )     # number of unique input values
        
        # Create the external spike generator
        self.poisson_noise = nest.Create('poisson_generator', params={'rate': self.ext_rate})
        # Create inhomogeneous poisson generator (time-dependent input signal)
        if self.input_s == 'signal':
            self.stim_pgen = nest.Create('inhomogeneous_poisson_generator', 1)
            # # external input (stimulus)
            self.sig, self.times, self.inp_times, self.inp_amplitudes = generate_piecewise_constant_signal(seed=self.seed, num_steps=num_steps, 
                                                                                    step_duration=step_duration, 
                                                                                    resolution=self.dt, scale=scale)
            
            self.stim_pgen.set({'rate_times': self.inp_times, 'rate_values': self.inp_amplitudes})
            nest.Connect(self.stim_pgen, self.neurons,
                        syn_spec = {'synapse_model': 'static_synapse',
                                    'delay': 1.5,
                                    'weight': self.w})
            
        # Connect the external drive to the neurons
        nest.Connect(self.poisson_noise, self.neurons,
                     syn_spec = {'synapse_model': 'static_synapse',
                                 'delay': 1.5,
                                 'weight': self.w})


    def simulate(self, t_sim):
        # Simulate the network with specified 
        nest.Simulate(t_sim)