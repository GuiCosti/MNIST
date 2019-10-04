"""
main.py
~~~~~~~
Main class to execute the neural networks and test.
"""

# Files
import neural_network
import mnist_loader

def main():
    fnn_single_hidden_layer(30, 30, 10, 3.0, False)

# SGD Tests
def fnn_single_hidden_layer(hidden_neurons, epochs, mini_batch_size, learning_rate, training_feedback):
    """Feedforward neural network using stochastic gradient descent"""

    # Loads traninig data   
    training_data, validation_data, test_data = mnist_loader.load_data() 

    # Basic three-layers based neural network with 748 neurons on input layer,'hidden_neurons'
    # value as neurons on hidden layer, 10 neurons on output layer.
    net = neural_network.Network([784, hidden_neurons, 10]) 

    # Train the network with 'epochs' value epochs, 'mini_batch_size' digits as mini-batch size,
    # 'learning_rate' as learning rate and test feedback if 'training_feedback' is true.
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, test_data=test_data if training_feedback else None)


if __name__ == "__main__":
    main()
