import numpy as np

class ConnectomeLinear:
    """
    Connectome-constrained linear layer implementing:
    - Biological extreme sparsity (default 98.5%)
    - Heavy-tailed log-normal synaptic weight distribution
    - Dale's principle (80% excitatory, 20% inhibitory populations)
    """
    def __init__(self, in_features, out_features, sparsity=0.985, exc_ratio=0.8):
        self.in_features = in_features
        self.out_features = out_features
        self.sparsity = sparsity
        self.exc_ratio = exc_ratio
        
        # Binary structural connectivity mask
        self.mask = (np.random.rand(in_features, out_features) > sparsity).astype(np.float32)
        
        # Heavy-tailed log-normal distribution (Song et al., PLoS Bio 2005)
        raw_weights = np.random.lognormal(mean=-1.0, sigma=0.6, size=(in_features, out_features)).astype(np.float32)
        
        # Dale's principle: excitatory vs inhibitory partitioning
        n_exc = int(exc_ratio * in_features)
        signs = np.ones(in_features, dtype=np.float32)
        signs[n_exc:] = -1.0
        
        # Normalization constant for stable forward gain
        norm_factor = 0.4 / np.sqrt(max(1.0, (1.0 - sparsity) * in_features))
        self.weights = self.mask * raw_weights * signs[:, None] * norm_factor

    def forward(self, x):
        """Computes sparse forward pass."""
        return np.dot(x, self.weights)

    def count_active_synops(self, spike_vector):
        """Counts non-zero synaptic operations triggered by binary active spikes."""
        active_indices = np.where(spike_vector > 0)[0]
        if len(active_indices) == 0:
            return 0
        return int(np.count_nonzero(self.weights[active_indices]))


class SpikingLIFLayer:
    """
    Leaky Integrate-and-Fire (LIF) spiking neuron layer:
    V[t] = beta * V[t-1] * (1 - S[t-1]) + I_in[t]
    S[t] = Theta(V[t] - V_th)
    """
    def __init__(self, n_units, beta=0.85, v_thresh=1.0, v_reset=0.0):
        self.n_units = n_units
        self.beta = beta
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.voltage = np.zeros(n_units, dtype=np.float32)
        self.spikes = np.zeros(n_units, dtype=np.float32)

    def reset(self):
        self.voltage = np.zeros(self.n_units, dtype=np.float32)
        self.spikes = np.zeros(self.n_units, dtype=np.float32)

    def step(self, current_input):
        # Decay voltage, reset if spiked previously
        self.voltage = self.beta * self.voltage * (1.0 - self.spikes) + current_input
        # Emit binary spikes
        self.spikes = (self.voltage >= self.v_thresh).astype(np.float32)
        return self.spikes


class ConnectomeNetwork:
    """
    Complete Bio-Inspired Neuromorphic Agent incorporating:
    Sensory Input -> Antennal Lobe Projection -> Recurrent Connectome SNN -> Motor Output
    """
    def __init__(self, n_in, n_hidden, n_out, sparsity=0.985):
        self.in_proj = np.random.randn(n_in, n_hidden) * 0.25
        self.recurrent_connectome = ConnectomeLinear(n_hidden, n_hidden, sparsity=sparsity)
        np.fill_diagonal(self.recurrent_connectome.weights, 0.0) # No autapses
        
        self.lif = SpikingLIFLayer(n_hidden)
        self.out_proj = np.random.randn(n_hidden, n_out) * 0.15
        
        self.total_synops = 0

    def reset_state(self):
        self.lif.reset()
        self.total_synops = 0

    def forward_step(self, x):
        I_in = np.dot(x, self.in_proj)
        
        # Recurrent biological transmission
        spikes_prev = self.lif.spikes
        active_ops = self.recurrent_connectome.count_active_synops(spikes_prev)
        self.total_synops += active_ops
        
        I_rec = self.recurrent_connectome.forward(spikes_prev)
        
        # Spiking integration
        spikes_curr = self.lif.step(I_in + I_rec)
        
        # Readout
        motor_out = np.tanh(np.dot(spikes_curr, self.out_proj))
        self.total_synops += np.count_nonzero(spikes_curr) * self.out_proj.shape[1]
        
        return motor_out
