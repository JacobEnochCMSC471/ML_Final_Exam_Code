# polynomial function
import numpy as np
import tensorflow as tf


def make_input_matrix():
    input_matrix_func = []

    while True:
        current_row = []
        raw_row_data = input('Enter a row of numbers (values separated by spaces) --> Q to exit\n')

        if raw_row_data == 'Q' or raw_row_data == 'q':
            break

        current_row_str = raw_row_data.split(" ")

        for data in current_row_str:
            temp = [int(data)]
            current_row.append(temp)

        input_matrix_func.append(current_row)

    return np.array(input_matrix_func)


x_in = [make_input_matrix()]


# Use in case the input method doesn't work for some reason
'''
x_in = np.array([
    [
  [[3], [0], [1], [2], [7], [4]],
  [[1], [5], [8], [9], [3], [1]],
  [[2], [7], [2], [5], [1], [3]],
  [[0], [1], [3], [1], [7], [8]],
  [[4], [2], [1], [6], [2], [8]],
  [[2], [4], [5], [2], [3], [9]]
    ]
])'''

input_matrix = tf.constant(x_in, dtype=tf.float32)

# Filter - layered in rows eg:
#  3 4 4
#  1 0 2
# -1 0 3

# Vertical edge detecting filter, 3x3
'''kernel_in = np.array([
 [[[1, 0.1]], [[0, 0.2]], [[-1, 0.3]]],
 [[[1, 0.4]], [[0, 0.5]], [[-1, 0.6]]],
 [[[1, 0.7]], [[0, 0.8]], [[-1, 0.9]]]
])'''

# 2x2 filter for easy copy/paste - just need to edit values
'''
kernel_in = np.array([
 [[[0, 0.1]], [[1, 0.2]]],
 [[[2, 0.4]], [[3, 0.5]]]
])'''

# 3x3 filter for easy copy/paste - just need to edit values

kernel_in = np.array([
    [[[7, 0.1]], [[9, 0.2]], [[-1, 0.3]]],
    [[[2, 0.4]], [[0, 0.5]], [[1, 0.6]]],
    [[[5, 0.7]], [[4, 0.8]], [[-2, 0.9]]]
])

# 5x5 filer for easy copy/paste - just need to edit values
'''
kernel_in = np.array([
    [[[3, 0.1]], [[4, 0.2]], [[4, 0.3]], [[4, 0.4]], [[4, 0.5]]],
    [[[3, 0.6]], [[4, 0.7]], [[4, 0.8]], [[4, 0.9]], [[4, 1.0]]],
    [[[3, 1.1]], [[4, 1.2]], [[4, 1.3]], [[4, 1.4]], [[4, 1.5]]],
    [[[3, 1.6]], [[4, 1.7]], [[4, 1.8]], [[4, 1.9]], [[4, 2.0]]],
    [[[3, 2.1]], [[4, 2.2]], [[4, 2.3]], [[4, 2.4]], [[4, 2.5]]],
])'''


padding = None

pad = input('How much padding? Enter Valid or 0 for no padding, Same for Same padding: \n')

if pad == 'Valid' or pad == 'valid' or pad == 0:  # No padding = Valid padding (Working)
    padding = 'VALID'

elif pad == 'Same' or pad == 'same':  # Output dims == Input dims is Same padding (Working)
    padding = 'SAME'

else:  # An integer was entered; padding is the integer (Working)
    pad = int(pad)
    padding = [[0, 0], [pad, pad], [pad, pad], [0, 0]]  # only edit middle values: [[0,0] [pad_top, pad_bottom], [pad_left, pad_right] [0,0]]

# padding = 1 adds 1 layer of zeroes to each side of the matrix
# padding = 2 adds 2 layers of zeroes to each side of the matrix, etc

kernel = tf.constant(kernel_in, dtype=tf.float32)

yes_no = input('Are the vertical and horizontal strides different (Y/N):\n')

if yes_no == 'Y' or yes_no == 'y':
    h_stride = int(input('Horizontal Stride: '))
    v_stride = int(input('Vertical Stride: '))
    res = tf.nn.conv2d(input_matrix, kernel, strides=[1, v_stride, h_stride, 1], padding=padding)

else:
    stride = int(input('Enter stride: \n'))

    # strides = [1, vertical, horizontal, 1]

    res = tf.nn.conv2d(input_matrix, kernel, strides=[1, stride, stride, 1], padding=padding)
    # padding = 'VALID -> no padding (use if specific padding is specified too, eg padding = 1 and uncomment line 65, padding=padding)
    # padding = 'SAME' -> same as input dims

print(res)
