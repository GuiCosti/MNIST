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
    """Defining hyper-parameters can be a hard task. To do so, there are some heuristics
    that can help.
    
    Broad Strategy: When using neural networks to attack a new problem the first challenge
    is to get any non-trivial learning, i.e., for the network to achieve results better than
    chance. Try to speedup your network feedback, by doing an sample space reducing.
    Ex1: Get rid of all the training and validation images except images which are 0s or 1s.
    Then try to train a network to distinguish 0s from 1s.
    Ex2: You can get another speed up in experimentation by increasing the frequency of monitoring.
    We can get feedback more quickly by monitoring the validation accuracy more often, say, after 
    every 1,000 training images."""

    ### Board Strategy ###

    # Reducing training set to 1.000
    net = neural_network_improved.Improved_Network([784, 10])
