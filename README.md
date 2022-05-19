# ML_Final_Exam_Code
Code that was used for calculations for the UMBC Spring 2022 Machine Learning final exam

final_code.py is used to calculate a convolution (operation used in convolutional neural networks). It asks a user to enter the values of the input matrix one row at a time
in a space-separated format (eg: 1 2 3 4 5 for the first row of as 5x5 matrix). It will also ask for the amount of padding, whether the vertical and horizontal strides
are different and ask for those respective strides if they are different, and if not, the general stride value. The Tensorflow tensorflow.nn.conv2 function is used to calculate
the convolution after all inputs are entered by the user. The results are then displayed in the console.

conv_output_dims.py calculates the output of a convolution given the input size, the filter size, stride and padding using the formula:

output_dims = floor(((input_size + (2 * padding_size) - filter_size) / stride)) + 1

activation_functions.py is simply a conglomeration of various NN-related functions, most of them being activation functions. It also has a description of 
some of the conditons for activation functions (such as SELU) and use-cases, drawbacks and strengths. 
