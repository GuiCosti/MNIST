"""
main.py
~~~~~~~
Main class to execute the neural networks and test.
"""

# Files
import neural_network
import mnist_loader

# SGD Tests
def sgd_tests():
    training_data, validation_data, test_data = mnist_loader.load_data() # Loads traninig data
    net = neural_network.Network([784, 30, 10]) # Basic three-layers based neural network with 748 neurons on input layer, 30 neurons on hidden layer, 10 neurons on output layer
    net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data) # train the network with 30 epochs, 10 digits as mini-batch size, 3.0 as learning rate and test feedback every epoch


if __name__ == "__main__":
    sgd_tests()