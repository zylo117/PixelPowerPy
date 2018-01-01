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

    plt.plot(diag_data, color="b", label=legend_a)
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
        plt.plot(i, color="b", label=legend_a)
    plt.xlim([0, len(diag_data_list[0])])
    plt.ylim([0.2, 1.2])

    plt.show()


def draw_diag_illumination_list_compare(diag_data_list_a, back_diag_data_list_a, diag_data_list_b,
                                        back_diag_data_list_b, x_size=512, y_range=[0.4, 0.6], legend_a=None, legend_b=None):
    plt.figure()
    plt.suptitle("Diagonal Illumination Distribution")
    plt.subplots_adjust(wspace=0.35, hspace=0.45)
    # plt.subplot(211), plt.plot(diag_data), plt.title('Diagonal'), plt.xlim([0, len(diag_data)])
    # plt.subplot(212), plt.plot(back_diag_data), plt.title('Back - Diagonal'), plt.xlim([0, len(diag_data)])

    # loc(location) code
    # 0,2 9   1
    # 6   10  7,5
    # 3   8   4

    for i in diag_data_list_a:
        plt.subplot(221)
        plt.plot(i, color="b", label=legend_a)
        plt.xlim([0, x_size])
        plt.ylim(y_range)
        plt.title('Top-Left')
        plt.xlabel("Pixel")
        plt.ylabel("Relative Illumination")
        plt.legend(loc=4)

        plt.subplot(224)
        plt.plot(i, color="b", label=legend_a)
        plt.xlim([len(diag_data_list_a[0]) - x_size, len(diag_data_list_a[0])])
        plt.ylim(y_range)
        plt.title('Bottom-Right')
        plt.xlabel("Pixel")
        plt.ylabel("Relative Illumination")
        plt.legend(loc=3)

    for i in back_diag_data_list_a:
        plt.subplot(222)
        plt.plot(i, color="b", label=legend_a)
        plt.xlim([len(back_diag_data_list_a[0]) - x_size, len(back_diag_data_list_a[0])])
        plt.ylim(y_range)
        plt.title('Top-Right')
        plt.xlabel("Pixel")
        plt.ylabel("Relative Illumination")
        plt.legend(loc=3)

        plt.subplot(223)
        plt.plot(i, color="b", label=legend_a)
        plt.xlim([0, x_size])
        plt.ylim(y_range)
        plt.title('Bottom-Left')
        plt.xlabel("Pixel")
        plt.ylabel("Relative Illumination")
        plt.legend(loc=4)

    for i in diag_data_list_b:
        plt.subplot(221)
        plt.plot(i, color="r", label=legend_b)
        plt.xlim([0, x_size])
        plt.ylim(y_range)
        plt.legend(loc=4)

        plt.subplot(224)
        plt.plot(i, color="r", label=legend_b)
        plt.xlim([len(diag_data_list_a[0]) - x_size, len(diag_data_list_a[0])])
        plt.ylim(y_range)
        plt.legend(loc=3)

    for i in back_diag_data_list_b:
        plt.subplot(222)
        plt.plot(i, color="r", label=legend_b)
        plt.xlim([len(back_diag_data_list_b[0]) - len(back_diag_data_list_b[0]) // 8, len(back_diag_data_list_b[0])])
        plt.ylim(y_range)
        plt.legend(loc=3)

        plt.subplot(223)
        plt.plot(i, color="r", label=legend_b)
        plt.xlim([0, len(back_diag_data_list_b[0]) // 8])
        plt.ylim(y_range)
        plt.legend(loc=4)

    plt.show()


def addarray2csv(data, output_path):
    f = open(output_path, 'a', newline="\n")
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()
