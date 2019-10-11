"""
main.py
~~~~~~~
Main class to execute the neural networks and test.
"""

# Files
import neural_network
import mnist_loader
import support_vector_machine as svm
from constant import Training_Dataset

def main():
    ### FNN (SGD) Tests ###
    fnn_single_hidden_layer(30, 30, 10, 3.0, True, Training_Dataset.Test_Data)


    ### SVM Tests ###
    #  svm.svm_baseline()

# SGD Tests
def fnn_single_hidden_layer(hidden_neurons, epochs, mini_batch_size, learning_rate, training_feedback, training_dataset_type: Training_Dataset):
    """Feedforward neural network using stochastic gradient descent."""

    # Loads traninig data   
    training_data, validation_data, test_data = mnist_loader.load_data()

    # Select the training dataset
    if(training_dataset_type == Training_Dataset.Validation_Data):
        training_dataset = validation_data
    elif(training_dataset_type == Training_Dataset.Test_Data):
        training_dataset = test_data

    # Basic three-layers based neural network with 748 neurons on input layer,'hidden_neurons'
    # value as neurons on hidden layer, 10 neurons on output layer.
    net = neural_network.Network([784, hidden_neurons, 10]) 

    # Train the network with 'epochs' value epochs, 'mini_batch_size' digits as mini-batch size,
    # 'learning_rate' as learning rate and test feedback if 'training_feedback' is true.
    print_information("FNN - Quadratic Cost", hidden_neurons, epochs, mini_batch_size, learning_rate)
    net.stochastic_gradient_descent(training_data, epochs, mini_batch_size, learning_rate, test_data=training_dataset if training_feedback else None)

# Miscellaneous
def print_information(neural_network: str, hidden_neurons: int, epochs: int, mini_batch_size: int, learning_rate: float):
    """Print out the information about the starting neural network"""
    print(F"Started: {neural_network} (Hidden Neurons: {hidden_neurons} | Epochs: {epochs} | Learning Rate {learning_rate})")

if __name__ == "__main__":
    main()
