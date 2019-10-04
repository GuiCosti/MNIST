"""
mnist_img_viewer.py
~~~~~~~~~~~~~~~~~~~
A library to to render MNIST images (28x28) with 748 gray-scale pixes using matplotlib
"""

# Third-party Libraries
import numpy as np
import matplotlib.pyplot as plt
import constant

def renderImage(img):
    """Receives an numpy array containing 784 gray-scale
    and print it with matplotlib."""

    if(str(img.shape) != constant.MNIST_IMAGE_SHAPE): # reshape the 784 gray-scale array to an image format 
        img = np.reshape(img, (-1, 28))

    plt.imshow(img, cmap='Greys')
    plt.show()