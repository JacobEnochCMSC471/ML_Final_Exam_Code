import numpy as np
import tensorflow as tf
from scipy.special import erfc


# polynomial function
def poly(x):
    return x ** 4 + 3 * x ** 2 + 7 * x + 3


# tanh function
def tanh(x):
    return np.tanh(x)


# ------- Gradient functions ------- #

# polynomial function
def d_poly(x):
    return 4 * x ** 3 + 6 * x + 7


# polinomial function
def d_tanh(x):
    h = tanh(x)
    return 1 - h ** 2


# ------- NN Activation Functions ------- #
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# RELU - good default activation function
def relu(z):
    return np.maximum(0, z)


# Smooth variant of ReLU - close to 0 when z is negative, close to z when z is positive
def soft_plus(z):
    return np.log(1 + np.exp(z))


# Derivative
def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps)) / (2 * eps)


# Leaky RELU
# To solve dying neurons problem, you may want to use a variant of the ReLU function, such as the leaky ReLU.
def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)


# ELU
'''
The ELU activation function looks a lot like the ReLU function, with a few major differences:

    ELU takes on negative values when z < 0, which allows the unit to have an average output closer to 0 and helps alleviate the vanishing gradients problem.

    The hyperparameter α defines the value that the ELU function approaches when z is a large negative number.
        
        It is usually set to 1, but you can tweak it like any other hyperparameter.
        
        It has a nonzero gradient for z < 0, which avoids the dead neurons problem.

    If α is equal to 1 then the function is smooth everywhere, including around z = 0, which helps speed up Gradient Descent since it does not bounce as much to the left and right of z = 0.

    The main drawback of the ELU activation function is that it is slower to compute than the ReLU function and its variants (due to the use of the exponential function).

    Its faster convergence rate during training compensates for that slow computation, but still, at test time an ELU network will be generally slower than a ReLU network.
'''


def elu(z, alpha=1):
    return np.where(z < 0, alpha * (np.exp(z) - 1), z)


alpha_0_1 = -np.sqrt(2 / np.pi) / (erfc(1 / np.sqrt(2)) * np.exp(1 / 2) - 1)

scale_0_1 = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (
        2 * erfc(np.sqrt(2)) * np.e ** 2 + np.pi * erfc(1 / np.sqrt(2)) ** 2 * np.e - 2 * (2 + np.pi) * erfc(1 / np.sqrt(2)) * np.sqrt(
    np.e) + np.pi + 2) ** (-1 / 2)

# SELU
'''
There are a few conditions for self-normalization of SELU to happen:

    The input features must be standardized (mean 0 and standard deviation 1).

    Every hidden layer’s weights must be initialized with LeCun normal initialization. In Keras, this means setting kernel_initializer="lecun_normal".

    The network’s architecture must be sequential.

    If you try to use SELU in nonsequential architectures, such as RNNs or networks with skip connections (i.e., connections that skip layers, such as in Wide & Deep nets), self-normalization will not be guaranteed, so SELU will not necessarily outperform other activation functions.

    The paper only guarantees self-normalization if all layers are dense, but some researchers have noted that the SELU activation function can improve performance in Convolutional Neural Nets (CNNs) as well

'''


def selu(z, scale=scale_0_1, alpha=alpha_0_1):
    return scale * elu(z, alpha)


print(tanh(-0.3))

