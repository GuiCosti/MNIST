"""
cost_functions.py
~~~~~~~~~~~~~~
Library containing the cost funcitions (Quadratic Cost and Cross Entropy Cost)
for improved neural networks.
"""

# Third-party libraries
import numpy as np

class QuadraticCost(object):
    """The Quadratic Cost is the function used on the basic neural network (neural_network.py).
    The problem with this function is that when the weights are saturated (the weight value is
    too far from it should be) the function learns much more slowly. This is because of the sigmoid 
    function, and it's graph shows us that when the artificial neuron is badly wrong, it has a lot of
    difficulty on learning."""
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output 'a' and desired output 'y'."""
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * sigmoid_prime(z)

class CrossEntropyCost(object):
    """The Cross-Entropy is an alternative cost function for our network, that solves the problem
    of learning slowdown. It learns fast when the neuron is saturated and also learns fast when it's
    nearly on the expected result. When should we use the cross-entropy instead of the quadratic cost?
    In fact, the cross-entropy is nearly always the better choice, provided the output neurons are sigmoid
    neurons"""
    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output 'a' and desired output
        'y'.  Note that np.nan_to_num is used to ensure numerical
        stability. In particular, if both 'a' and 'y' have a 1.0
        in the same slot, then the expression (1-y)*np.log(1-a)
        returns nan.  The np.nan_to_num ensures that that is converted
        to the correct value (0.0)."""
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))