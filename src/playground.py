"""
playground.py
~~~~~~~
Class for user's experimentation and training (in non neural network context).
"""

import neural_network_improved
import cost_functions as cf
import mnist_loader

def playground():
    """Method for user's experimentation and training (in non neural network context)"""
    hyper_parameters()

def hyper_parameters():
    """Defining hyper-parameters can be a hard task. To do so, there are some heuristics that can help."""

    # Load Traning datasets
    training_data, validation_data, test_data = mnist_loader.load_data()
    training_data = list(training_data)
    validation_data = list(validation_data)
    test_data = list(test_data)


    # Board Strategy
    # board_strategy(training_data, validation_data, test_data)

    # Learning Rate Strategy
    # learning_rate_strategy(training_data, validation_data, test_data)


def board_strategy(training_data, validation_data, test_data):
    """Broad Strategy: When using neural networks to attack a new problem the first challenge
    is to get any non-trivial learning, i.e., for the network to achieve results better than
    chance. Try to speedup your network feedback, by doing an sample space reducing.
    Ex1: Get rid of all the training and validation images except images which are 0s or 1s.
    Then try to train a network to distinguish 0s from 1s.
    Ex2: You can get another speed up in experimentation by increasing the frequency of monitoring.
    We can get feedback more quickly by monitoring the validation accuracy more often, say, after 
    every 1,000 training images."""

    net = neural_network_improved.Improved_Network([784, 10])

    # Reducing training set to 1.000 and only 0 and 1 images
    # reduced_training_set_and_numbers_one_zero(training_data, validation_data, net)

    # Reducing training set to 1.000
    reduced_training_set(training_data, validation_data, net)
                                

def reduced_training_set_and_numbers_one_zero(training_data, validation_data, net: neural_network_improved.Improved_Network):
    """Reduces training set to 1.000 images containing only numbers '0' and '1'."""
    training_data = list(filter(lambda img: img[1][0] == 1 or img [1][1] == 1, training_data)) # Filter traninig set to contains only "0" and "1" numbers images
    validation_data = list(filter(lambda img: img[1] == 1 or img [1] == 0, validation_data)) # Filter validations set to contains only "0" and "1" numbers images
    net.stochastic_gradient_descent(training_data[:1000],
                                30,
                                20,
                                40.0,
                                lmbda = 1.0,
                                evaluation_data=validation_data[:100],
                                monitor_evaluation_accuracy=True)

def reduced_training_set(training_data, validation_data, net: neural_network_improved.Improved_Network):
    """Reduces training set to 1.000 images."""
    net.stochastic_gradient_descent(training_data[:1000],
                                    30,
                                    50,
                                    0.1,
                                    lmbda=0.1,
                                    evaluation_data=validation_data[:100],
                                    monitor_evaluation_accuracy=True)

def learning_rate_strategy(training_data, validation_data, test_data):
    """Learning Rate Strategy: To get a good learning rate parameter, we need to find a value that constantly
    deacreses over the epocs. Choosing a small value can cause a slow stochastic gradiente descent problem as
    well. A good approach is to start with an greater value, to go down fast into the ideal value, and than
    start to decrease the leaning rate so it can't 'climb up' the function graph. The first approach that can
    be used is to find the threshold where the learning rate starts decreasing. Following this procedure will
    give us an order of magnitude estimate for the threshold value of learning rate. The values of learning
    rate you should use should SMALLER than your threshold value."""

    net = neural_network_improved.Improved_Network([784, 10])

    threshold_learning_rate(training_data, validation_data, net, 0.025) # Learning rate = 0.025
    threshold_learning_rate(training_data, validation_data, net, 0.25) # Learning rate = 0.25
    threshold_learning_rate(training_data, validation_data, net, 2.5) # Learning rate = 2.5

def threshold_learning_rate(training_data, validation_data, net: neural_network_improved.Improved_Network, learning_rate):
    """Apply the stochastic gradient descent with various learning rates"""
    net.stochastic_gradient_descent(training_data,
                            30,
                            10,
                            learning_rate,
                            lmbda = 1.0,
                            evaluation_data=validation_data,
                            monitor_evaluation_accuracy=True)