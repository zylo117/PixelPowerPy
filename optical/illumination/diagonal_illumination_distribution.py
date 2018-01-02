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
from regression_tool import *


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
    # x = np.linspace(0, len(diag_data) - 1, len(diag_data))
    # param = np.polyfit(x, diag_data, deg=24)
    # y = np.polyval(param, x)
    # der = np.polyder(param)
    # y_der = np.polyval(der, x)
    # plt.plot(x, y, color="r")
    plt.xlim([0, len(diag_data)])
    plt.ylim([0.2, 1.2])

    # plt.figure()
    # plt.plot(x, y_der, color="g")

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


def draw_diag_illumination_list_compare(diag_data_list_a, back_diag_data_list_a, diag_data_list_b,
                                        back_diag_data_list_b, x_size=512, y_range=[0.4, 0.6], legend_a=None,
                                        legend_b=None, polyFit=False, polyDer=0):
    # loc(location) code
    # 0,2 9   1
    # 6   10  7,5
    # 3   8   4

    # Calculate Regression Value
    if polyDer != 0:
        plt.figure()
        if polyDer == 1:
            plt.suptitle("Derivative of Diagonal Illumination Distribution")
        if polyDer == 2:
            plt.suptitle("Inflection Point of Diagonal Illumination Distribution")
        plt.subplots_adjust(hspace=0.35)
        plt.subplot(211)
        plt.title("Diagonal")
        plt.xlim([0, len(diag_data_list_a[0])])
        # center line
        plt.plot(np.arange(len(diag_data_list_a[0])), np.zeros(len(diag_data_list_a[0])), color="y", linestyle="--")

        plt.subplot(212)
        plt.title("Back-Diagonal")
        plt.xlim([0, len(diag_data_list_a[0])])
        plt.legend(loc="1")
        # center line
        plt.plot(np.arange(len(diag_data_list_a[0])), np.zeros(len(diag_data_list_a[0])), color="y", linestyle="--")

    if polyFit:
        for i in range(len(diag_data_list_a)):
            x = np.linspace(0, len(diag_data_list_a[i]) - 1, len(diag_data_list_a[i]))

            param = np.polyfit(x, diag_data_list_a[i], deg=24)
            diag_data_list_a[i] = np.polyval(param, x)
            if polyDer != 0:
                plt.subplot(211)
                der = np.polyder(param, m=polyDer)
                y_der = np.polyval(der, x)
                if i == 0:
                    plt.plot(x, y_der, color="b", label=legend_a)
                else:
                    plt.plot(x, y_der, color="b")


            param = np.polyfit(x, back_diag_data_list_a[i], deg=24)
            back_diag_data_list_a[i] = np.polyval(param, x)
            if polyDer != 0:
                plt.subplot(212)
                der = np.polyder(param, m=polyDer)
                y_der = np.polyval(der, x)
                if i == 0:
                    plt.plot(x, y_der, color="b", label=legend_a)
                else:
                    plt.plot(x, y_der, color="b")


        for i in range(len(diag_data_list_b)):
            x = np.linspace(0, len(diag_data_list_b[i]) - 1, len(diag_data_list_b[i]))

            param = np.polyfit(x, diag_data_list_b[i], deg=24)
            diag_data_list_b[i] = np.polyval(param, x)
            if polyDer != 0:
                plt.subplot(211)
                der = np.polyder(param, m=polyDer)
                y_der = np.polyval(der, x)
                if i == 0:
                    plt.plot(x, y_der, color="r", label=legend_b)
                else:
                    plt.plot(x, y_der, color="r")

            param = np.polyfit(x, back_diag_data_list_b[i], deg=24)
            back_diag_data_list_b[i] = np.polyval(param, x)
            if polyDer != 0:
                plt.subplot(212)
                der = np.polyder(param, m=polyDer)
                y_der = np.polyval(der, x)
                if i == 0:
                    plt.plot(x, y_der, color="r", label=legend_b)
                else:
                    plt.plot(x, y_der, color="r")
    if polyDer != 0:
        plt.legend(loc="1")

    plt.figure()
    plt.suptitle("Diagonal Illumination Distribution")
    plt.subplots_adjust(wspace=0.35, hspace=0.45)

    for i in range(len(diag_data_list_a)):
        plt.subplot(221)
        if i == 0:
            plt.plot(diag_data_list_a[i], color="b", label=legend_a)
        else:
            plt.plot(diag_data_list_a[i], color="b")
        plt.xlim([0, x_size])
        plt.ylim(y_range)
        plt.title('Top-Left')
        plt.xlabel("Pixel")
        plt.ylabel("Relative Illumination")

        plt.subplot(224)
        if i == 0:
            plt.plot(diag_data_list_a[i], color="b", label=legend_a)
        else:
            plt.plot(diag_data_list_a[i], color="b")
        plt.xlim([len(diag_data_list_a[0]) - x_size, len(diag_data_list_a[0])])
        plt.ylim(y_range)
        plt.title('Bottom-Right')
        plt.xlabel("Pixel")
        plt.ylabel("Relative Illumination")

    for i in range(len(back_diag_data_list_a)):
        plt.subplot(222)
        if i == 0:
            plt.plot(back_diag_data_list_a[i], color="b", label=legend_a)
        else:
            plt.plot(back_diag_data_list_a[i], color="b")
        plt.xlim([len(back_diag_data_list_a[0]) - x_size, len(back_diag_data_list_a[0])])
        plt.ylim(y_range)
        plt.title('Top-Right')
        plt.xlabel("Pixel")
        plt.ylabel("Relative Illumination")

        plt.subplot(223)
        if i == 0:
            plt.plot(back_diag_data_list_a[i], color="b", label=legend_a)
        else:
            plt.plot(back_diag_data_list_a[i], color="b")
        plt.xlim([0, x_size])
        plt.ylim(y_range)
        plt.title('Bottom-Left')
        plt.xlabel("Pixel")
        plt.ylabel("Relative Illumination")

    for i in range(len(diag_data_list_b)):
        plt.subplot(221)
        if i == 0:
            plt.plot(diag_data_list_b[i], color="r", label=legend_b)
        else:
            plt.plot(diag_data_list_b[i], color="r")
        plt.xlim([0, x_size])
        plt.ylim(y_range)

        plt.subplot(224)
        if i == 0:
            plt.plot(diag_data_list_b[i], color="r", label=legend_b)
        else:
            plt.plot(diag_data_list_b[i], color="r")
        plt.xlim([len(diag_data_list_a[0]) - x_size, len(diag_data_list_a[0])])
        plt.ylim(y_range)

    for i in range(len(back_diag_data_list_b)):
        plt.subplot(222)
        if i == 0:
            plt.plot(back_diag_data_list_b[i], color="r", label=legend_b)
        else:
            plt.plot(back_diag_data_list_b[i], color="r")
        plt.xlim([len(back_diag_data_list_b[0]) - len(back_diag_data_list_b[0]) // 8, len(back_diag_data_list_b[0])])
        plt.ylim(y_range)

        plt.subplot(223)
        if i == 0:
            plt.plot(back_diag_data_list_b[i], color="r", label=legend_b)
        else:
            plt.plot(back_diag_data_list_b[i], color="r")
        plt.xlim([0, len(back_diag_data_list_b[0]) // 8])
        plt.ylim(y_range)

    # set legend
    plt.subplot(221)
    plt.legend(loc=4)
    plt.subplot(222)
    plt.legend(loc=3)
    plt.subplot(223)
    plt.legend(loc=4)
    plt.subplot(224)
    plt.legend(loc=3)

    plt.show()


def fit_curve(diag_data):
    max_intensity = np.max(diag_data)
    length = len(diag_data)
    # illuminance_curve_param(length, max_intensity)
    popt, pcov = illuminance_curvefit(np.arange(length), diag_data)
    return popt, pcov


def draw_curve(diag_data, curve_data):
    plt.figure()
    plt.title("Diagonal Illumination Regression Distribution")
    plt.xlabel("Pixel")
    plt.ylabel("Relative Illumination")
    # plt.subplot(211), plt.plot(diag_data), plt.title('Diagonal'), plt.xlim([0, len(diag_data)])
    # plt.subplot(212), plt.plot(back_diag_data), plt.title('Back - Diagonal'), plt.xlim([0, len(diag_data)])

    plt.plot(diag_data, color="b")
    plt.plot(curve_data, color="g")
    plt.xlim([0, len(diag_data)])
    plt.ylim([0.2, 1.2])

    plt.show()


def addarray2csv(data, output_path):
    f = open(output_path, 'a', newline="\n")
    writer = csv.writer(f)
    writer.writerow(data)
    f.close()
