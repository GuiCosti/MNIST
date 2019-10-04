"""
mnist_loader.py
~~~~~~~~~~~~~~~~
A library to load the MNIST data for usage. The method __load_mnist_data__() returns the pure data structures.
The load_data() method returns the data structure in a neural network friendly structure, so basic you should use this one.
For more information, read the method's descriptions.
"""

# Standart Libraries
import pickle # Package to serialize and deserialize binary objects to text or vice-versa
import gzip # Unzip library (7z like)

# Third-party Libraries
import numpy as np

def __load_mnist_data__():
    """Return the pure MNIST data as tuple containing the training data,
    validation data and the test data.
    
    The 'training_data' is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.
    
    The second entry in the 'training_data' tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    
    The 'validation_data' and 'test_data' are similar, except
    each contains only 10,000 images."""

    # Fixing problem caused by the difference in encoding codec used by Python3s pickle
    with gzip.open('../data/mnist.pkl.gz','rb') as ff :
        u = pickle._Unpickler( ff )
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load() # Load dataset

    return (training_data, validation_data, test_data)

def load_data():
    """Returns the MNIST data with a more convenient structure for using on neural networks.
    
    The __load_mnist_data__() loads the pure MNIST data that contains 50,000 tuples (x, y),
    where 'x' is a 748-dimensional numpy.drray containing the input image and 'y' is a 10-dimensional
    numpy.darray containing the corresponding classification as a digit value (integers).
    
    This methods converts this 'y' value into a a 10-dimensional numpy.ndarray representing the
    unit vector corresponding to the correct digit for 'x' (only on 'training_data' case). Also turns the 748-dimensional numpy.drray
    diposed on columns, to become diposed as rows"""
    tr_d, va_d, te_d = __load_mnist_data__() # Loads pure data

    # Turns the 748-dimensional array disposed as columns, to become disposed as rows. Simple change the vector orientation
    # ex: [0, 0, 0] -> [[0],
    #                   [0],
    #                   [0]]
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorize_to_ten_dimension(y) for y in tr_d[1]] # Change the answer from digit value based to 10-dimensional vector answer based
    training_data = zip(training_inputs, training_results) # Creates a new array containing the 748-dimensional array and the 10-dimensional answer array

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return (training_data, validation_data, test_data)

def vectorize_to_ten_dimension(digit):
    """Return a 10-dimensional unit vector with a 1.0 in the digit
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1)) # generates a 10-dimensional numpy.ndarray with 0's. The parameters of np.zeros are (number of rows, number of columns)
    e[digit] = 1.0 # set the answer on 10-dimensional array
    return e