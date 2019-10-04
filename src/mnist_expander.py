"""
mnist_expander.py
~~~~~~~~~~~~~~~~~~
This file contains the code to expand the MNIST dataset, slightly moving the 50.000 images in all directions (ex: one time up, than down, etc),
so it helps the neural network to not get biased on the image at that exactly position, creating a set of 250.000 images.
It loads the dataset and than creates a new file called mnist_expanded.pkl.gz with the slightly moved images (for 1 image, another 4 are created)
"""

# Standart Libraries
import pickle # Package to serialize and deserialize binary objects to text or vice-versa
import os.path # Managing system paths
import gzip # Unzip library (7z like)
import random

# Third-party Libraries
import numpy as np

# Files
import mnist_img_viewer as mnistimgview

print("Expanding Mnist Dataset")

def changeImagePosition(image, answer):
    """ Slightly move the image in all directions (ex: one time up, than down, etc),
    so it helps the neural network to not get biased on the image at that exactly
    position. Here you can use mnistimgview.renderImage() to see the slightly
    movement of the images. It will generate 4 images instead of 1 image that was inputed.
    the answer is the same for all images"""

    moved_images = []
    for d, axis, index_position, index in [
        (1,  0, "first", 0),
        (-1, 0, "first", 27),
        (1,  1, "last",  0),
        (-1, 1, "last",  27)]:
        new_img = np.roll(image, d, axis)
        if index_position == "first": 
            new_img[index, :] = np.zeros(28)
        else: 
            new_img[:, index] = np.zeros(28)
        moved_images.append((np.reshape(new_img, 784), answer))
    return moved_images


if (os.path.exists("../data/mnist_expanded.pkl.gz")):
    print("The expanded training set 'mnist_expanded.pkl.gz' already exists on expecified path.\nExiting.")
else:
    # Fixing problem caused by the difference in encoding codec used by Python3s pickle
    with gzip.open('../data/mnist.pkl.gz','rb') as ff :
        u = pickle._Unpickler( ff )
        u.encoding = 'latin1'
        training_data, validation_data, test_data = u.load() # Load dataset

    expanded_training_pairs = []

    j = 0 # counter
    # Here the function is zipping the array(training_data) that is a tuple,
    # the first position (training_data[0]) containing the 784 gray-scale pixels (ex: [0, 0.28, 0 ... 0])
    # and the second one containing the answer (training_data[1]) (ex: 5)
    for x, y in zip(training_data[0], training_data[1]):
        expanded_training_pairs.append((x, y)) # add this [pixels , answer] array to a variable
        # mnistimgview.renderImage(x) # render the pixel vector as image
        image = np.reshape(x, (-1, 28)) # reshape image to 28x28
        j += 1

        if j % 1000 == 0: print(F"Expanded {j} images...", j)

        # Get the current image and moves it to all directions (up, down, left write)
        # creating 4 new images, slightly moved from the original center, so it can help
        # the neural network to no be biased on the image at that certain position.
        slightly_moved_images = changeImagePosition(image, y)

        for x in slightly_moved_images:
           expanded_training_pairs.append(x) 

    random.shuffle(expanded_training_pairs) # shuffle all images

    expanded_training_data = [list(d) for d in zip(*expanded_training_pairs[:30000])] # Unzip part of the expanded_training_pairs (not all because it was causing memory leak)
    
    print("Saving expanded data. This may take a few minutes.")

    f = gzip.open("../data/mnist_expanded.pkl.gz", "w") # Write the new dataset with expanded images
    pickle.dump((expanded_training_data, validation_data, test_data), f) # Turns the text into binary
    f.close()
    
    print("Expanded dataset created successfully.")