import numpy as np


def array2bin(imagedata, inputformat="raw"):
    if inputformat is "raw":
        height, width = imagedata.shape
        output1d = np.reshape(imagedata, (height * width))
        output1d = np.array((width, height, output1d))

    elif inputformat is "bayer":
        height, width, c = imagedata.shape
        height *= 2
        width *= 2
        raw_imagedata = np.zeros((height, width))
        raw_imagedata[::2, ::2] = imagedata[:, :, 0]
        raw_imagedata[::2, 1::2] = imagedata[:, :, 1]
        raw_imagedata[1::2, ::2] = imagedata[:, :, 2]
        raw_imagedata[1::2, 1::2] = imagedata[:, :, 3]

        output1d = np.reshape(raw_imagedata, (height * width))
        output1d = np.hstack((width, height, output1d))
        print(0)

    return 0
