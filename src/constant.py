"""
constant.py
~~~~~~~~~
Library of constant values
"""

from enum import Enum, unique

### Constants ###

ZIP_TYPE = "<class 'zip'>"
MNIST_IMAGE_SHAPE = "(28, 28)"


### Enums ###

@unique
class Training_Dataset(Enum):
    Unknown = 0
    Test_Data = 1
    Validation_Data = 2

@unique
class Cost_Function(Enum):
    Unknown = 0
    Quadratic_Cost = 1
    Cross_Entropy = 2
    