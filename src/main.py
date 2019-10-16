"""
main.py
~~~~~~~
Main class to execute the neural networks and test.
"""

# Files
import neural_network
import neural_network_improved
import cost_functions as cf
import mnist_loader
import support_vector_machine as svm
from constant import Training_Dataset
from constant import Cost_Function

def main():
    ### FNN (SGD) Tests ###
    # fnn(30, 30, 10, 3.0, True, Training_Dataset.Test_Data) 
    # fnn(30, 30, 10, 3.0, True, Training_Dataset.Validation_Data) # validation_data contains 10,000 images of digits, different from the 50,000 images in the MNIST training set, and  MNIST test set (prevent overfitting)

    ### Improved FNN (Cross-Entropy, L2 Regularization, Weights Initialization) ###
    improved_fnn(30, 30, 10, 0.1, 5.0, Training_Dataset.Validation_Data, monitor_evaluation_accuracy= True)


    ### SVM Tests ###
    #  svm.svm_baseline()

# SGD Tests
def fnn(hidden_neurons, epochs, mini_batch_size, learning_rate, training_feedback, training_dataset_type: Training_Dataset):
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

def improved_fnn(hidden_neurons,
                epochs,
                mini_batch_size,
                learning_rate,
                lmbda = 5.0,
                training_dataset_type: Training_Dataset = Training_Dataset.Test_Data,
                cost_function_type: Cost_Function = Cost_Function.Cross_Entropy,
                monitor_evaluation_cost=False,
                monitor_evaluation_accuracy=False,
                monitor_training_cost=False,
                monitor_training_accuracy=False):
    """Improved Feedforward Neural Network using stochastic gradient descent.
    Improved using Cross-Entropy, L2 Regularization, Weights Initialization technics."""

    # Loads traninig data   
    training_data, validation_data, test_data = mnist_loader.load_data()

    # Select the training dataset
    if(training_dataset_type == Training_Dataset.Validation_Data):
        training_dataset = validation_data
    elif(training_dataset_type == Training_Dataset.Test_Data):
        training_dataset = test_data

    # Improved three-layers based neural network with 748 neurons on input layer,'hidden_neurons'
    # value as neurons on hidden layer, 10 neurons on output layer.
    net = neural_network_improved.Improved_Network([784, hidden_neurons, 10]) 

    # Train the network with 'epochs' value epochs, 'mini_batch_size' digits as mini-batch size,
    # 'learning_rate' as learning rate and test feedback if 'training_feedback' is true.
    print_information("Improved FNN", hidden_neurons, epochs, mini_batch_size, learning_rate)
    net.stochastic_gradient_descent(training_data, 
                                    epochs,
                                    mini_batch_size,
                                    learning_rate,
                                    lmbda,
                                    training_dataset,
                                    monitor_evaluation_cost,
                                    monitor_evaluation_accuracy,
                                    monitor_training_cost,
                                    monitor_training_accuracy)



# Miscellaneous
def print_information(neural_network: str, hidden_neurons: int, epochs: int, mini_batch_size: int, learning_rate: float):
    """Print out the information about the starting neural network"""
    print(F"Started: {neural_network} (Hidden Neurons: {hidden_neurons} | Epochs: {epochs} | Learning Rate {learning_rate})")

if __name__ == "__main__":
    main()
