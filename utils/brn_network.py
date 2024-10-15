"""
Helper functions for creating simple Brunel networks.
"""

from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
import nest
import itertools
import sys


class BrunelNetwork:
    def __init__(
        self,
        num_neurons,
        rho,
        eps,
        g,
        eta,
        J,
        delay,
        neuron_params,
        n_rec_ex,
        n_rec_in,
        # rec_start,
        # rec_stop,
    ):
        """

        Parameters
        ----------
        num_neurons
        rho
        eps
        g
        eta
        J
        delay
        neuron_params
        n_rec_ex
        n_rec_in
        """
        self.num_neurons = num_neurons
        self.num_ex = int((1 - rho) * num_neurons)  # number of excitatory neurons
        self.num_in = int(rho * num_neurons)  # number of inhibitory neurons
        self.c_ex = int(eps * self.num_ex)  # number of excitatory connections
        self.c_in = int(eps * self.num_in)  # number of inhibitory connections
        self.J_ex = J  # excitatory weight
        self.J_in = -g * J  # inhibitory weigh
        self.delay = delay
        self.n_rec_ex = n_rec_ex  # number of recorded excitatory neurons, both excitatory and inhibitory
        self.n_rec_in = n_rec_in  # number of recorded excitatory neurons, both excitatory and inhibitory
        # self.rec_start = rec_start
        # self.rec_stop = rec_stop
        self.neuron_params = neuron_params  # neuron params
        self.ext_rate = (
            self.neuron_params["V_th"] / (J * self.c_ex * self.neuron_params["tau_m"]) * eta * 1000.0 * self.c_ex
        )

    def create(self, syn_spec=None):
        # Create the network
        if syn_spec is None:
            syn_spec = {"synapse_model": "static_synapse"}

        # First create the neurons
        self.neurons_ex = nest.Create("iaf_psc_delta", self.num_ex, params=self.neuron_params)
        self.neurons_in = nest.Create("iaf_psc_delta", self.num_in, params=self.neuron_params)

        self.neurons = self.neurons_ex + self.neurons_in

        # Then create the external spike generator
        self.poisson_noise = nest.Create("poisson_generator", params={"rate": self.ext_rate})

        # Then create spike detectors
        self.detector_ex = nest.Create("spike_recorder", self.n_rec_ex)
        self.detector_in = nest.Create("spike_recorder", self.n_rec_in)

        # Next we connect the neurons
        nest.Connect(
            self.neurons_ex,
            self.neurons_ex,
            conn_spec={"rule": "fixed_indegree", "indegree": self.c_ex},
            syn_spec=syn_spec | {"delay": self.delay, "weight": self.J_ex},
        )
        nest.Connect(
            self.neurons_ex,
            self.neurons_in,
            conn_spec={"rule": "fixed_indegree", "indegree": self.c_ex},
            syn_spec={"synapse_model": "static_synapse", "delay": self.delay, "weight": self.J_ex},
        )

        nest.Connect(
            self.neurons_in,
            self.neurons,
            conn_spec={"rule": "fixed_indegree", "indegree": self.c_in},
            syn_spec={"synapse_model": "static_synapse", "delay": self.delay, "weight": self.J_in},
        )

        # Then we connect the external drive to the neurons
        nest.Connect(
            self.poisson_noise,
            self.neurons,
            syn_spec={"synapse_model": "static_synapse", "delay": self.delay, "weight": self.J_ex},
        )

        # Then we connect the the neurons to the spike detectors
        nest.Connect(self.neurons_ex[: self.n_rec_ex], self.detector_ex, "one_to_one")
        nest.Connect(self.neurons_in[: self.n_rec_in], self.detector_in, "one_to_one")

    def simulate(self, t_sim):
        # Simulate the network with specified
        nest.Simulate(t_sim)

    def get_spikes_concatenated(self):
        """
        Compiles the spiking activity of E/I neurons into dictionaries with "senders" and "spike times".

        Returns
        -------

        """
        spikes_ex, spikes_in = self.get_data()
        res_ex = defaultdict(list)
        res_in = defaultdict(list)

        for i, spikes in enumerate(spikes_ex):
            res_ex["senders"].extend([i] * len(spikes))
            res_ex["times"].extend(spikes)

        for i, spikes in enumerate(spikes_in):
            res_in["senders"].extend([i] * len(spikes))
            res_in["times"].extend(spikes)

        return res_ex, res_in

    def get_data(self):
        # get spikes from recorders
        spikes_ex = []
        spikes_in = []

        for i in range(self.n_rec_ex):
            spikes_ex.append(list(np.sort(nest.GetStatus(self.detector_ex)[i]["events"]["times"])))
        for i in range(self.n_rec_in):
            spikes_in.append(list(np.sort(nest.GetStatus(self.detector_in)[i]["events"]["times"])))

        return spikes_ex, spikes_in


# Helper function to plot spiking activity
def plot_raster_rate(spikes_ex, spikes_in, rec_start, rec_stop, figsize=(9, 5)):
    spikes_ex_total = list(itertools.chain(*spikes_ex))
    spikes_in_total = list(itertools.chain(*spikes_in))
    spikes_total = spikes_ex_total + spikes_in_total

    n_rec_ex = len(spikes_ex)
    n_rec_in = len(spikes_in)

    fig = plt.figure(constrained_layout=True, figsize=figsize)
    gs = fig.add_gridspec(5, 1)

    ax1 = fig.add_subplot(gs[:4, 0])
    ax2 = fig.add_subplot(gs[4, 0])

    ax1.set_xlim([rec_start, rec_stop])
    ax2.set_xlim([rec_start, rec_stop])

    ax1.set_ylabel("Neuron ID")

    ax2.set_ylabel("Firing rate")
    ax2.set_xlabel("Time [ms]")

    for i in range(n_rec_in):
        ax1.plot(
            spikes_in[i],
            i * np.ones(len(spikes_in[i])),
            linestyle="",
            marker="o",
            color="r",
            markersize=2,
        )
    for i in range(n_rec_ex):
        ax1.plot(
            spikes_ex[i],
            (i + n_rec_in) * np.ones(len(spikes_ex[i])),
            linestyle="",
            marker="o",
            color="b",
            markersize=2,
        )

    ax2 = ax2.hist(spikes_ex_total, range=(rec_start, rec_stop), bins=int(rec_stop - rec_start))

    # plt.savefig('raster.png')
    plt.show()

    time_diff = (rec_stop - rec_start) / 1000.0
    average_firing_rate = len(spikes_total) / time_diff / (n_rec_ex + n_rec_in)
    print(f"Average firing rate: {average_firing_rate} Bq")
