"""
diagonal illumination distribution of the image
detect Lens shading defects
detect if defect shading corner(s) exist(s)

INPUT:
  IDraw: raw image data
  bayerFormat: bayer format:
      'bggr'
      'rggb'
      'grbg'
      'gbrg'

OUTPUT:
  diagonal and back-diagonal illumination distribution of the image, 2 series of float number
"""

from io_bin.preprocess import preprocess
import numpy as np
import matplotlib.pyplot as plt
import csv


def di(ID, bayerformat="rggb", pedestal=64, bitdepth=10, custom_source=None):
    if custom_source is None:
        IDyuv = preprocess(ID, bayerformat=bayerformat, pedestal=pedestal, bitdepth=bitdepth, outputformat="yuv")
    else:
        IDyuv = custom_source

    IDy = IDyuv[:, :, 0]

    h, w = IDy.shape

    slope = h / w
    slope_back = - slope

    diag = []
    back_diag = []

    for i in range(w):
        # diag
        y = (slope * i)
        big = int(np.ceil(y))
        little = int(y)
        if big != little:
            diag.append(np.mean(IDy[little: big + 1, i]))
        else:
            diag.append(np.mean(IDy[little, i]))

        # back-diag
        y_back = (slope_back * i + h)
        big_back = int(np.ceil(y_back))
        little_back = int(y_back)
        if big != little:
            back_diag.append(np.mean(IDy[little_back: big_back + 1, i]))
        else:
            back_diag.append(np.mean(IDy[little, i]))

    # normalize the data
    diag = np.array(diag)
    back_diag = np.array(back_diag)

    diag = diag / np.max(diag)
    back_diag = back_diag / np.max(back_diag)

    return diag, back_diag


def draw_diag_illumination(diag_data, back_diag_data):
    plt.figure()
    plt.title("Diagonal Illumination Distribution")
    plt.xlabel("Pixel")
    plt.ylabel("Relative Illumination")
    # plt.subplot(211), plt.plot(diag_data), plt.title('Diagonal'), plt.xlim([0, len(diag_data)])
    # plt.subplot(212), plt.plot(back_diag_data), plt.title('Back - Diagonal'), plt.xlim([0, len(diag_data)])

    plt.plot(diag_data, color="b")
    plt.plot(back_diag_data, color="g")
    plt.xlim([0, len(diag_data)])
    plt.ylim([0.2, 1.2])

    plt.show()

def draw_diag_illumination_list(diag_data_list):
    plt.figure()
    plt.title("Diagonal Illumination Distribution")
    plt.xlabel("Pixel")
    plt.ylabel("Relative Illumination")
    # plt.subplot(211), plt.plot(diag_data), plt.title('Diagonal'), plt.xlim([0, len(diag_data)])
    # plt.subplot(212), plt.plot(back_diag_data), plt.title('Back - Diagonal'), plt.xlim([0, len(diag_data)])

    for i in diag_data_list:
        plt.plot(i, color="b")
    plt.xlim([0, len(diag_data_list[0])])
    plt.ylim([0.2, 1.2])

    plt.show()

def draw_diag_illumination_list_compare(diag_data_list_a, diag_data_list_b):
    plt.figure()
    plt.title("Diagonal Illumination Distribution")
    plt.xlabel("Pixel")
    plt.ylabel("Relative Illumination")
    # plt.subplot(211), plt.plot(diag_data), plt.title('Diagonal'), plt.xlim([0, len(diag_data)])
    # plt.subplot(212), plt.plot(back_diag_data), plt.title('Back - Diagonal'), plt.xlim([0, len(diag_data)])

    for i in diag_data_list_a:
        plt.plot(i, color="b")

    for i in diag_data_list_b:
        plt.plot(i, color="r")

    plt.xlim([0, len(diag_data_list_a[0])])
    plt.ylim([0.2, 1.2])

    plt.show()

def addarray2csv(data, output_path):
    f = open(output_path, 'a', newline="\n")
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()
