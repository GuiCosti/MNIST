"""
network_network.py
~~~~~~~~~~
A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.
"""

# Standard library
import random

# Third-party libraries
import numpy as np

# File
import constant

class Network(object):
    def __init__(self, sizes):
        """The list 'sizes' contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution (or Normal Distribution) with mean 0, and variance 1.
        Note that the first layer is assumed to be an input layer,
        and by convention we won't set any biases for those neurons,
        since biases are only ever used in computing the outputs
        from later layers. Note that 'numpy.random.randn()' 
        return a sample (or samples) from the 'standard normal'
        distribution and the parameters are the dimensions."""
        self.num_layers = len(sizes) # number of layers (ex: three-layer network)
        self.sizes = sizes # number of neurons in the respective layers (shape of the neural network)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]] # biases of each layer (Note that the Network initialization code assumes that the first layer of neurons is an input layer, and omits to set any biases for those neurons)
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])] # '[:-1]' slices the string to omit the last character. '[1:]' slices the string to omit the first character
        
    def feedforward(self, a):
        """Return the output of the network if 'a' is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, learning_rate, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent (SGD). The 'training_data' is a list of tuples
        '(x, y)' representing the training inputs and the desired
        outputs. The other non-optional parameters are
        self-explanatory. If 'test_data' is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out. This is useful for
        tracking progress, but slows things down substantially."""
        
        training_data, test_data = convert_zip_data(training_data, test_data) # Fixing python 3 problem with zip returning object

        if test_data: n_test = len(test_data) # Gets the test_data length if exists (Note: list() is forcing a zip-type variable to evaluate so it has an len() value)
        n = len(training_data) # Training_data length (Note: list() is forcing a zip-type variable to evaluate so it has an len() value)
        for j in range(epochs): # Iterate over the epochs
            random.shuffle(training_data) # shuffle the training data to avoid bias
            mini_batches = [ # Creates the mini-batches based on mini_batch_size, from the training_data
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches: # Iterate over the mini-batches
                self.update_mini_batch(mini_batch, learning_rate) # Update the current mini-batch weights and biases based on learning rate
            if test_data: # Gives or not the test feedback based on test_data state
                current_evaluation = self.evaluate(test_data)
                print (F"Epoch {j}: {current_evaluation} / {n_test} Correct Classifications - Accuracy " + "{:.1%}".format(current_evaluation/n_test))
            else:
                print (F"Epoch {j} complete")
    
    def update_mini_batch(self, mini_batch, learning_rate):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The 'mini_batch' is a list of tuples '(x, y)'"""
        new_biases = [np.zeros(b.shape) for b in self.biases] # Creates a new empty vector store the new biases values
        new_weights = [np.zeros(w.shape) for w in self.weights] # Creates a new empty vector store the new weights values
        for x, y in mini_batch: # Iterate over values in this mini-batch
            delta_b, delta_w = self.backprop(x, y) # Apply the backpropagation logic to get ΔC
            new_biases = [nb+dnb for nb, dnb in zip(new_biases, delta_b)]
            new_weights = [nw+dnw for nw, dnw in zip(new_weights, delta_w)]
        self.weights = [w - (learning_rate/len(mini_batch)) * nw # Apply the rule to move the weights to the local minimum (v -> v' = v - η∇C)
                        for w, nw in zip(self.weights, new_weights)]
        self.biases = [b - (learning_rate/len(mini_batch)) * nb # Apply the rule to move the biases to the local minimum (v -> v' = v - η∇C)
                       for b, nb in zip(self.biases, new_biases)]

    def backprop(self, x, y):
        """Return a tuple '(new_biases, new_weights)' representing the
        gradient for the cost function C_x.  'new_biases' and
        'new_weights' are layer-by-layer lists of numpy arrays, similar
        to 'self.biases' and 'self.weights'."""
        new_biases = [np.zeros(b.shape) for b in self.biases]
        new_weights = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # List to store all the smoothed (by the sigmoid function) activations, layer by layer
        zs = [] # List to store all the z vectors, layer by layer (z = σ(w⋅x+b)), where x is the neuron input; w is the neuron weight; b is the neuron bias and σ is the sigmoid function
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # np.dot() is a matrix multiplication of w⋅x+b
            zs.append(z) # Add Z value on z array
            activation = sigmoid(z) # set the activation value as sigmoid(z), smoothing activation value
            activations.append(activation) # Add smoothed activation on activations array
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) # Calculates the derivate of the variation
        new_biases[-1] = delta
        new_weights[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            new_biases[-l] = delta
            new_weights[-l] = np.dot(delta, activations[-l-1].transpose())
        return (new_biases, new_weights)
    
    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \\partial C_x \\partial a for the output activations."""
        return (output_activations-y)

#### Misc Functions
def sigmoid(z):
    """The sigmoid function. It's smooth the neuron result
    to create small changes on the output. Instead of being
    just 0 or 1, these inputs can also take on any values 
    between 0 and 1. So, for instance, 0.638… is a valid 
    input for a sigmoid neuron. This sigmoid function could
    be exchanged for ReLU (Rectified Linear Unit) function 
    as well."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def convert_zip_data(training_data, test_data):
    training_data = extract_zip_list(training_data)
    test_data = extract_zip_list(test_data)
    return (training_data, test_data)

def extract_zip_list(obj):
    """In Python 2, zip returned a list. In Python 3, zip returns an iterable object.
    But you can make it into a list just by calling list. This method do this."""
    if(str(type(obj)) == constant.ZIP_TYPE):
        return list(obj)
    return obj
        