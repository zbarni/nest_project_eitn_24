# NEST project

## 2024/10/17

Mercredi on bloque pour préparer la présentation.

Options :
* weight, noise, neuron basic parameters
* Clustering of the network...
* Plasticity (based model is static) => Short-Term Plasticity, STDP ?
* Different neuron model (model with adaptation...? usefull to study memory at the single neuron level, spike-frequency adaptation ?)
* Encoding layer (translate stimulus into a spiking activity)
* Decoding layer

We don't have to create our model but we can if we wan't. The decoding part is the metric. 

Encoding layer to transform stimulus in spike activity :
* effect of the duration (20ms, try to go tà 70-80ms)
* influence of differents parameters and
* 100 neurons on the population, 10% connectivity, scaled_weight by the squared root of ten (follow the stimulus), g always 5 that mean that the inhibitory is 5 times bigger than the excitatory

$$C(u,y) = \frac{cov(u,y)^2}{var(u) \times var(y)}$$