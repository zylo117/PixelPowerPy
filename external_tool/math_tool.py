import numba
import numpy as np
from scipy.ndimage.filters import convolve


def conv2(x, y, mode='same'):
    """
    Emulate the function conv2 from Mathworks.

    Usage:

    z = conv2(x,y,mode='same')

    TODO:
     - Support other modes than 'same' (see conv2.m)
    """

    if not (mode == 'same'):
        raise Exception("Mode not supported")

    # Add singleton dimensions
    if (len(x.shape) < len(y.shape)):
        dim = x.shape
        for i in range(len(x.shape), len(y.shape)):
            dim = (1,) + dim
        x = x.reshape(dim)
    elif (len(y.shape) < len(x.shape)):
        dim = y.shape
        for i in range(len(y.shape), len(x.shape)):
            dim = (1,) + dim
        y = y.reshape(dim)

    origin = ()

    # Apparently, the origin must be set in a special way to reproduce
    # the results of scipy.signal.convolve and Matlab
    for i in range(len(x.shape)):
        if ((x.shape[i] - y.shape[i]) % 2 == 0 and
                x.shape[i] > 1 and
                y.shape[i] > 1):
            origin = origin + (-1,)
        else:
            origin = origin + (0,)

    z = convolve(x, y, mode='constant', origin=origin)

    return z


@numba.jit()
def imfilter_with_1d_kernel(in_array, kernel, axis=0):
    length_kernel = len(kernel)
    length_side = length_kernel // 2

    # 水平滤波
    if axis == 0:
        row = in_array.shape[0]
        col = in_array.shape[1] - 2 * length_side
        output = np.zeros((row, col))

        for i in range(row):
            for j in range(length_side, col + length_side):
                sum = 0
                for k in range(length_kernel):
                    sum = sum + kernel[k] * in_array[i][j - length_side + k]

                output[i][j - length_side] = sum

    # 竖直滤波
    if axis == 1:
        row = in_array.shape[0] - 2 * length_side
        col = in_array.shape[1]
        output = np.zeros((row, col))

        for i in range(length_side, row + length_side):
            for j in range(col):
                sum = 0
                for k in range(length_kernel):
                    sum = sum + kernel[k] * in_array[i - length_side + k][j]

                output[i - length_side][j] = sum

    return output


def rescale_intensity(in_array, targetval_max=255, threshold=0, dtype=np.uint8):
    max = np.max(in_array)
    if threshold == 0:
        factor = targetval_max / max
    else:
        factor = targetval_max / threshold
    return (in_array * factor).astype(dtype)


def point_distance(point1, point2):
    return np.sqrt((point1[1] - point2[1]) ** 2 + (point1[0] - point2[0]) ** 2)
