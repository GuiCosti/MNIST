"""
neural_network_improved.py
~~~~~~~~~~~~~~
An improved version of network.py, implementing the stochastic
gradient descent learning algorithm for a feedforward neural network.

Improvements include the addition of the cross-entropy cost function,
regularization, and better initialization of network weights.
"""

# Standard library
import json
import random
import sys

# Third-party libraries
import numpy as np

# Files
import cost_functions as cf

class Improved_Network(object):
    
    def __init__(self, sizes, cost=cf.CrossEntropyCost):
        """The list 'sizes' contains the number of neurons in the respective
        layers of the network. Ex: if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using a Gaussian
        distribution (or Normal Distribution) with mean 0, and variance 1.
        Note that the first layer is assumed to be an input layer,
        and by convention we won't set any biases for those neurons,
        since biases are only ever used in computing the outputs
        from later layers."""