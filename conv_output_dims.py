import numpy as np


# Calculates the output size for a convolutional neural network's feature map, given input dimensions, filter size, padding size and stride
def get_output_dim_size():
    input_size = int(input('What is the input matrix dimension? (NxN):\n'))
    filter_size = int(input('What is the size of the filter? (FxF):\n'))
    padding_size = int(input('What is the padding size?:\n'))
    conv_stride = int(input('What is the stride?:\n'))

    output_dim = np.floor(((input_size + (2 * padding_size) - filter_size) / conv_stride)) + 1
    output_dim = int(output_dim)

    return 'Feature map dimensions: {}x{}'.format(output_dim, output_dim)


print(get_output_dim_size())
