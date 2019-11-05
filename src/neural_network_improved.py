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
import constant

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
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.improved_weight_initializer()
        self.cost = cost
        self.best_score = 0
    
    def improved_weight_initializer(self):
        """Improved initialization with each weight using a Gaussian 
        distribution (Normal distribution) with mean 0 and standard deviation
        1 over the square root of the number of weights connecting to the same
        neuron. Initialize the biases using a Gaussian distribution with mean 0
        and standard deviation 1.

        This improved initiation sharps the normal distribution, making the
        standard deviation smaller than the other neural network. Such approach
        makes the neurons much less likely to saturate, and correspondingly much
        less likely to have problems with a learning slowdown.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers."""
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]] # biases of each layer (Note that the Network initialization code assumes that the first layer of neurons is an input layer, and omits to set any biases for those neurons)
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])] # strategy: 1/sqrt(Number of inputs) - 
                        #'[:-1]' slices the string to omit the last character. '[1:]' slices the string to omit the first character.

    def feedforward(self, a):
        """Return the output of the network if 'a' is input."""
        for b, w in zip(self.biases, self.weights):
            a = cf.sigmoid(np.dot(w, a)+b)
        return a

    def stochastic_gradient_descent(self,
            training_data,
            epochs,
            mini_batch_size,
            learning_rate,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False):
        """Train the neural network using mini-batch stochastic gradient
        descent.  The 'training_data' is a list of tuples '(x, y)'
        representing the training inputs and the desired outputs. The
        other non-optional parameters are self-explanatory, as is the
        regularization parameter 'lmbda' (lambda). The method also accepts
        'evaluation_data', usually either the validation or test
        data. We can monitor the cost and accuracy on either the
        evaluation data or the training data, by setting the
        appropriate flags.
        
        The method returns a tuple containing four
        lists: the (per-epoch) costs on the evaluation data, the
        accuracies on the evaluation data, the costs on the training
        data, and the accuracies on the training data. All values are
        evaluated at the end of each training epoch. So, for example,
        if we train for 30 epochs, then the first element of the tuple
        will be a 30-element list containing the cost on the
        evaluation data at the end of each epoch. Note that the lists
        are empty if the corresponding flag is not set."""

        training_data, evaluation_data = convert_zip_data(training_data, evaluation_data) # Fixing python 3 problem with zip returning object

        if evaluation_data: n_data = len(evaluation_data) # check if exists an evaluation_data.
        n = len(training_data) # attribute the length of training data to n.
        evaluation_cost, evaluation_accuracy = [], [] # create 2 arrays to hold evaluation cost and accuracy
        training_cost, training_accuracy = [], [] # create 2 arrays to hold training cost and accuracy

        for j in range(epochs): # iterate over the epochs

            random.shuffle(training_data) # shuffle the training data to avoid biases on networking training
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] # creates the mini-batches based on mini_batch_size, from the training_data

            for mini_batch in mini_batches: # iterate over the mini-batches
                self.update_mini_batch(mini_batch, learning_rate, lmbda, len(training_data)) # update the current mini-batch weights and biases based on learning rate
            print (F"Epoch {j} training complete")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print (F"Cost on training data: {cost}")

            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print (F"Accuracy on training data: {accuracy} / {n} (" + "{:.1%}".format(accuracy/n) + ")")

            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print (F"Cost on evaluation data: {cost}")

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print (F"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {n_data} (" + "{:.1%}".format(accuracy/n_data) + ")")
            print
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    def stochastic_gradient_descent_early_stopping(
            self,
            training_data,
            mini_batch_size,
            learning_rate,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            no_improvements_in_epochs=10):
        """Modified stochastic gradiente descent method that uses the
        Early Stopping Strategy to determinate the 'best' number of epochs.
        Basically, the 'no_improvements_in_epochs' parameter is going to
        determinate the number of epochs that you going to monitor before
        stopping to train the network."""

        training_data, evaluation_data = convert_zip_data(training_data, evaluation_data) # Fixing python 3 problem with zip returning object

        if evaluation_data: n_data = len(evaluation_data) # check if exists an evaluation_data.
        n = len(training_data) # attribute the length of training data to n.
        evaluation_cost, evaluation_accuracy = [], [] # create 2 arrays to hold evaluation cost and accuracy
        training_cost, training_accuracy = [], [] # create 2 arrays to hold training cost and accuracy
        epochs_since_last_best_score = 0 # this number will keep the number of epochs since the last best score
        epoch = 1 # epochs iterator

        while epoch < 10000: # iterate over the epochs (Note: 10.000 was created for not running infinitely while discovering the best number of epochs)

            random.shuffle(training_data) # shuffle the training data to avoid biases on networking training
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] # creates the mini-batches based on mini_batch_size, from the training_data

            for mini_batch in mini_batches: # iterate over the mini-batches
                self.update_mini_batch(mini_batch, learning_rate, lmbda, len(training_data)) # update the current mini-batch weights and biases based on learning rate
            print (F"Epoch {epoch} training complete")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print (F"Cost on training data: {cost}")

            # Training accuracy becomes required on this type of training
            accuracy = self.accuracy(training_data, convert=True)
            training_accuracy.append(accuracy)
            
            # Apply early stopping logic
            accuracy_rate = accuracy/n
            if (epochs_since_last_best_score == no_improvements_in_epochs):
                print(F"Best epochs hyper-parameter for current configuration: {epoch - no_improvements_in_epochs} | Best Score {accuracy_rate * 100}%")
                return epoch
            if (accuracy_rate <= self.best_score):
                epochs_since_last_best_score += 1
            else:
                self.best_score = accuracy_rate
                epochs_since_last_best_score = 0

            print (F"Accuracy on training data: {accuracy} / {n} (" + "{:.1%}".format(accuracy/n) + ")")
            print (F"Epochs since last best score: {epochs_since_last_best_score}")
        
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print (F"Cost on evaluation data: {cost}")

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print (F"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {n_data} (" + "{:.1%}".format(accuracy/n_data) + ")")
            print
            epoch += 1 # Add one epoch
        return epoch

    def stochastic_gradient_descent_learning_rate_schedule(
            self,
            training_data,
            mini_batch_size,
            learning_rate,
            lmbda = 0.0,
            evaluation_data=None,
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            no_improvements_in_epochs=4,
            learning_rate_reducing_factor=2,
            max_epochs=10000):
        """Modified stochastic gradiente descent method that uses the
        Learning Rate Schedule Strategy to improve the way learning rate
        affects the training. The idea is to hold the learning rate constant
        until the validation accuracy starts to get worse. Then decrease the
        learning rate by some amount, say a factor of two or ten. We repeat
        this many times, until, say, the learning rate is a factor of 1,024
        (or 1,000) times lower than the initial value. Then we terminate."""

        training_data, evaluation_data = convert_zip_data(training_data, evaluation_data) # Fixing python 3 problem with zip returning object

        if evaluation_data: n_data = len(evaluation_data) # check if exists an evaluation_data.
        n = len(training_data) # attribute the length of training data to n.
        evaluation_cost, evaluation_accuracy = [], [] # create 2 arrays to hold evaluation cost and accuracy
        training_cost, training_accuracy = [], [] # create 2 arrays to hold training cost and accuracy
        epochs_since_last_best_score = 0 # this number will keep the number of epochs since the last best score
        learning_rate_stopping_value = learning_rate / (learning_rate_reducing_factor ** 10) # learning rate 10 times lower than the initial value. It will be used to stop the algorithm
        epoch = 1 # epochs iterator

        while epoch < max_epochs: # iterate over the epochs (Note: 10.000 was created for not running infinitely while discovering the best number of epochs)

            random.shuffle(training_data) # shuffle the training data to avoid biases on networking training
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)] # creates the mini-batches based on mini_batch_size, from the training_data

            for mini_batch in mini_batches: # iterate over the mini-batches
                self.update_mini_batch(mini_batch, learning_rate, lmbda, len(training_data)) # update the current mini-batch weights and biases based on learning rate
            print (F"Epoch {epoch} training complete")

            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print (F"Cost on training data: {cost}")

            # Training accuracy becomes required on this type of training
            accuracy = self.accuracy(training_data, convert=True)
            training_accuracy.append(accuracy)
            
            # Learning rate scheduler logic
            accuracy_rate = accuracy/n
            if (learning_rate <= learning_rate_stopping_value):
                print(F"Best Score {accuracy_rate * 100}% with learning rate scheduler using factor: {learning_rate_reducing_factor}")
                return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
            if (accuracy_rate <= self.best_score):
                epochs_since_last_best_score += 1
            else:
                self.best_score = accuracy_rate
                epochs_since_last_best_score = 0
            if (epochs_since_last_best_score >= no_improvements_in_epochs):
                print(F"Learning rate changed to {round(learning_rate ** 1/2, 4)} (Previous: {round(learning_rate, 4)}) (Stopes on {round(learning_rate_stopping_value, 4)})")
                epochs_since_last_best_score = 0
                learning_rate = learning_rate / learning_rate_reducing_factor

            print (F"Accuracy on training data: {accuracy} / {n} (" + "{:.1%}".format(accuracy/n) + ")", F" - Epochs since last best score: {epochs_since_last_best_score}")
        
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print (F"Cost on evaluation data: {cost}")

            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print (F"Accuracy on evaluation data: {self.accuracy(evaluation_data)} / {n_data} (" + "{:.1%}".format(accuracy/n_data) + ")")
            print
            epoch += 1 # Add one epoch
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy

    def update_mini_batch(self, mini_batch, learning_rate, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch. The
        'mini_batch' is a list of tuples '(x, y)', 'learning_rate' is the
        learning rate, 'lmbda' is the regularization parameter, and
        'n' is the total size of the training data set."""

        new_biases = [np.zeros(b.shape) for b in self.biases] # creates a new empty vector store the new biases values
        new_weights = [np.zeros(w.shape) for w in self.weights] # creates a new empty vector store the new weights values
        for x, y in mini_batch: # iterate mini-batch
            delta_b, delta_w = self.backprop(x, y) # apply the backpropagation to figure out the difference between partial derivatives for biases (b) and weights (w) at the point C.
            new_biases = [nb+dnb for nb, dnb in zip(new_biases, delta_b)] 
            new_weights = [nw+dnw for nw, dnw in zip(new_weights, delta_w)]
        self.weights = [(1-learning_rate*(lmbda/n))*w-(learning_rate/len(mini_batch))*nw
                        for w, nw in zip(self.weights, new_weights)]  # apply the rule to move the weights to the local minimum, this time using lambda as well (v -> v' = v - η∇C). Also applying L2 regularization
        self.biases = [b-(learning_rate/len(mini_batch))*nb 
                       for b, nb in zip(self.biases, new_biases)] # apply the rule to move the biases to the local minimum (v -> v' = v - η∇C). Also applying L2 regularization

    def backprop(self, x, y):
        """Return a tuple '(new_biases, new_weights)' representing the
        gradient for the cost function C_x.  'new_biases' and
        'new_weights' are layer-by-layer lists of numpy arrays, similar
        to 'self.biases' and 'self.weights'."""

        new_biases = [np.zeros(b.shape) for b in self.biases]
        new_weights = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the smoothed activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer (z = σ(w⋅x+b)), where x is the neuron input; w is the neuron weight; b is the neuron bias and σ is the sigmoid function
        
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b # np.dot() is a matrix multiplication of w⋅x+b -> multiply all inputs for they respective weights and add their biases
            zs.append(z) # store the result above in Z array
            activation = cf.sigmoid(z) # set the activation value as sigmoid(z), smoothing activation value
            activations.append(activation) # add the smoothed activation to the activations array
        # backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y) # apply the Cost Delta Function 
        new_biases[-1] = delta # update biases with new delta
        new_weights[-1] = np.dot(delta, activations[-2].transpose()) # updated weights multiplying the activations 
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers): # iterate over network layers
            z = zs[-l] # attribute the zs of that layer to z. Uses a feature from Python to access the list backward. ex: l[-3] is the third last entry in a list l
            sp = cf.sigmoid_prime(z) # finds the prime derivative of the sigmoid function 
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            new_biases[-l] = delta
            new_weights[-l] = np.dot(delta, activations[-l-1].transpose())
        return (new_biases, new_weights)

    ### Monitor Functions
    def total_cost(self, data, lmbda, convert=False):
        """Return the total cost for the data set 'data'. The flag
        'convert' should be set to False if the data set is the
        training data (the usual case), and to True if the data set is
        the validation or test data.  See comments on the similar (but
        reversed) convention for the 'accuracy' method, below."""

        cost = 0.0 
        for x, y in data: # iterate over the data
            a = self.feedforward(x) # apply feedforward function over the inputs
            if convert: y = vectorize_to_ten_dimension(y) # convert a digit (0...9) into a corresponding desired output from the neural network.
            cost += self.cost.fn(a, y)/len(data)
        cost += 0.5*(lmbda/len(data))*sum(
            np.linalg.norm(w)**2 for w in self.weights)
        return cost
    
    def accuracy(self, data, convert=False):
        """Return the number of inputs in 'data' for which the neural
        network outputs the correct result. The neural network's
        output is assumed to be the index of whichever neuron in the
        final layer has the highest activation.

        The flag 'convert' should be set to False if the data set is
        validation or test data (the usual case), and to True if the
        data set is the training data. The need for this flag arises
        due to differences in the way the results 'y' are
        represented in the different data sets. In particular, it
        flags whether we need to convert between the different
        representations. It may seem strange to use different
        representations for the different data sets. Why not use the
        same representation for all three data sets? It's done for
        efficiency reasons -- the program usually evaluates the cost
        on the training data and the accuracy on other data sets.
        These are different types of computations, and using different
        representations speeds things up. More details on the
        representations can be found in mnist_loader.load_data()."""

        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y))
                       for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)


### Misc Functions

    def save(self, filename):
        """Save the neural network to the file 'filename'."""
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

def load(filename):
    """Load a neural network from the file 'filename'. Returns an
    instance of Network."""

    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Improved_Network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net

def vectorize_to_ten_dimension(digit):
    """Return a 10-dimensional unit vector with a 1.0 in the digit
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1)) # generates a 10-dimensional numpy.ndarray with 0's. The parameters of np.zeros are (number of rows, number of columns)
    e[digit] = 1.0 # set the answer on 10-dimensional array
    return e

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