
import numpy as np
import math

class Connections:
    """ Define connections between spiking neuron layers """

    def __init__(self, weights, pre_dimension, post_dimension):
        """
        Args:
            weights (ndarray): connection weights
            pre_dimension (int): dimension for pre-synaptic neurons
            post_dimension (int): dimension for post-synaptic neurons
        """
        self.weights = weights
        self.pre_dimension = pre_dimension
        self.post_dimension = post_dimension

    def __call__(self, spike_input):
        """
        Args:
            spike_input (ndarray): spikes generated by the pre-synaptic neurons
        Return:
            psp: postsynaptic layer activations
        """
        psp = np.matmul(self.weights, spike_input)
        return psp
    
class LIFNeurons:
    """
        Define Leaky Integrate-and-Fire Neuron Layer
        This class is complete. You do not need to do anything here.
    """

    def __init__(self, dimension, vdecay, vth):
        """
        Args:
            dimension (int): Number of LIF neurons in the layer
            vdecay (float): voltage decay of LIF neurons
            vth (float): voltage threshold of LIF neurons

        """
        self.dimension = dimension
        self.vdecay = vdecay
        self.vth = vth

        # Initialize LIF neuron states
        self.volt = np.zeros(self.dimension)
        self.spike = np.zeros(self.dimension)

    def __call__(self, psp_input):
        """
        Args:
            psp_input (ndarray): synaptic inputs
        Return:
            self.spike: output spikes from the layer
                """
        self.volt = self.vdecay * self.volt * (1. - self.spike) + psp_input
        self.spike = (self.volt > self.vth).astype(float)
        return self.spike
  

class SNN:
    """ Define a Spiking Neural Network with No Hidden Layer """

    def __init__(self, input_2_output_weight,
                 input_dimension=200, output_dimension=2,
                 vdecay=0.5, vth=0.5, snn_timestep=20):
        """
        Args:
            input_2_hidden_weight (ndarray): weights for connection between input and hidden layer
            hidden_2_output_weight (ndarray): weights for connection between hidden and output layer
            input_dimension (int): input dimension
            hidden_dimension (int): hidden_dimension
            output_dimension (int): output_dimension
            vdecay (float): voltage decay of LIF neuron
            vth (float): voltage threshold of LIF neuron
            snn_timestep (int): number of timesteps for inference
        """
        self.snn_timestep = snn_timestep
        self.output_layer = LIFNeurons(output_dimension, vdecay, vth)
        self.input_2_output_connection = Connections(input_2_output_weight, input_dimension, output_dimension)

    def __call__(self, spike_encoding):
        """
        Args:
            spike_encoding (ndarray): spike encoding of input
        Return:
            spike outputs of the network
        """
        spike_output = np.zeros(self.output_layer.dimension)
        for tt in range(self.snn_timestep):
            input_2_output_psp = self.input_2_output_connection(spike_encoding)
            output_spikes = self.output_layer(input_2_output_psp)
            spike_output += output_spikes
        return spike_output/self.snn_timestep
    

class STDP():
    """Train a network using STDP learning rule"""
    def __init__(self, network, A_plus, A_minus, tau_plus, tau_minus, lr, snn_timesteps=20, epochs=30, w_min=0, w_max=1):
        """
        Args:
            network (SNN): network which needs to be trained
            A_plus (float): STDP hyperparameter
            A_minus (float): STDP hyperparameter
            tau_plus (float): STDP hyperparameter
            tau_minus (float): STDP hyperparameter
            lr (float): learning rate
            snn_timesteps (int): SNN simulation timesteps
            epochs (int): number of epochs to train with. Each epoch is defined as one pass over all training samples.
            w_min (float): lower bound for the weights
            w_max (float): upper bound for the weights
        """
        self.network = network
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.snn_timesteps = snn_timesteps
        self.lr = lr
        self.time = np.arange(0, self.snn_timesteps, 1)
        self.sliding_window = np.arange(-4, 4, 1) #defines a sliding window for STDP operation.
        self.epochs = epochs
        self.w_min = w_min
        self.w_max = w_max

    def update_weights(self, t, i):
        """
        Function to update the network weights using STDP learning rule

        Args:
            t (int): time difference between postsynaptic spike and a presynaptic spike in a sliding window
            i(int): index of the presynaptic neuron

        Fill the details of STDP implementation
        """
        #compute delta_w for positive time difference
        if t>0:
          delta_w = self.A_plus * math.exp((-1) * t / self.tau_plus)

        #compute delta_w for negative time difference
        else:
          delta_w = (-1) * self.A_minus * math.exp((-1) * t / self.tau_minus)

        #update the network weights if weight increment is negative
        if delta_w < 0:
          self.network.input_2_output_connection.weights[:,i] += self.lr * delta_w * (self.network.input_2_output_connection.weights[:,i] - self.w_min)

        #update the network weights if weight increment is positive
        elif delta_w > 0:
          self.network.input_2_output_connection.weights[:,i] += self.lr * delta_w * (self.w_max - self.network.input_2_output_connection.weights[:,i])

    # def train_step(self, train_data_sample):
    #     """
    #     Function to train the network for one training sample using the update function defined above.

    #     Args:
    #         train_data_sample (list): a sample from the training data

    #     This function is complete. You do not need to do anything here.
    #     """
    #     input = train_data_sample[0]
    #     output = train_data_sample[1]
    #     for t in self.time:
    #         if output[t] == 1:
    #             for i in range(2):
    #                 for t1 in self.sliding_window:
    #                     if (0<= t + t1 < self.snn_timesteps) and (t1!=0) and (input[i][t+t1] == 1):
    #                         self.update_weights(t1, i)

    def train_step(self, train_data_sample):
        """
        Function to train the network for one training sample using the update function defined above.

        Args:
            train_data_sample (list): a sample from the training data
        """
        input, output = train_data_sample  # Assuming 'output' is similarly a 1D array for a single timestep
        if len(input) != 200:
            raise ValueError("Input data must be a 1D array of size 200")

        # Process output at t=0 (since you mentioned only one timestep)
        if output[0] == 1:  # Assuming output is a binary indicator of a spike
            for i in range(200):  # Go through each neuron
                for t1 in self.sliding_window:
                    t = 0  # Since timestep is 1, we consider only the current timestep '0'
                    if 0 <= t + t1 < self.snn_timesteps and t1 != 0:  # Check if time shift is within bounds
                        if input[i] == 1:  # Check if there is a spike in the neuron 'i'
                            self.update_weights(t1, i)


    def train(self, training_data):
        """
        Function to train the network

        Args:
            training_data (list): training data

        This function is complete. You do not need to do anything here.
        """
        for ee in range(self.epochs):
            for train_data_sample in training_data:
                self.train_step(train_data_sample)


class Learning:
    def hebbian(network, train_data, lr=1e-5, epochs=10):
        """
        Function to train a network using Hebbian learning rule
            Args:
                network (SNN): SNN network object
                lr (float): learning rate
                train_data (list): training data
                epochs (int): number of epochs to train with. Each epoch is defined as one pass over all training samples.

            Write the operations required to compute the weight increment according to the hebbian learning rule. Then increment the network weights.
        """
        print("data len: ", len(train_data))
        #iterate over the epochs
        for ee in range(epochs):
            #iterate over all samples in train_data
            for data in train_data:
                #compute the firing rate for the input
                input_data, output_data = data

                print("input data len: ", len(input_data))
                print("input data: ", input_data)
                print("output data: ", output_data)
                print("output data len: ", len(output_data))

                input_firing_rates = input_data

                print("input_firing_rates: ", input_firing_rates)

                #compute the firing rate for the output
                network_output = network(input_data)
                output_firing_rates = network_output

                #compute the correlation using the firing rates calculated above
                correlation_using_firing_rates = np.outer(output_firing_rates, input_firing_rates)

                #compute the weight increment
                weight_increment = correlation_using_firing_rates * lr

                #increment the weight
                network.input_2_output_connection.weights += weight_increment
