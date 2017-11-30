import array
import numpy
from functionset import *


def preprocess(imageinput, bayerformat="rggb", outputformat="raw", mode=0, bitdepth=10, pedestal=64, FOV=0,
               whitebalance=True, signed=True):
    # unsigned integer, 16位
    raw = array.array('H', open(imageinput, "rb").read())

    width = raw[0]
    height = raw[1]

    raw = numpy.array(raw)
    raw = raw[2:]

    raw = raw.reshape((height, width))
    print(raw[0][0], raw[0][1], raw[1][0], raw[1][1])

    # 按mode切片
    raw = crop_by_mode(raw, mode)

    # 增益
    raw = raw + pedestal
    if not signed:
        raw[raw < 0] = 0
    print(raw[0][0], raw[0][1], raw[1][0], raw[1][1])

    # 白平衡
    if whitebalance:
        raw = white_balance(raw)
        raw[raw > 2 ** bitdepth - 1] = 2 ** bitdepth - 1  # 防过饱和

    # 去镜头阴影
    if FOV is not 0:
        raw = lens_shading_correction(raw, 75)
        raw[raw > 2 ** bitdepth - 1] = 2 ** bitdepth - 1  # 防过饱和
    print(raw[0][0], raw[0][1], raw[1][0], raw[1][1])

    # 图像格式转换
    if outputformat is "raw":
        # 源格式输出
        return raw

    elif outputformat is "bayer":
        # 分4层输出，强制转换到RGrGbB
        plane = numpy.zeros((int(raw.shape[0] / 2), int(raw.shape[1] / 2), 4))
        R = numpy.zeros((int(raw.shape[0] / 2), int(raw.shape[1] / 2)))
        Gr = numpy.zeros((int(raw.shape[0] / 2), int(raw.shape[1] / 2)))
        Gb = numpy.zeros((int(raw.shape[0] / 2), int(raw.shape[1] / 2)))
        B = numpy.zeros((int(raw.shape[0] / 2), int(raw.shape[1] / 2)))

        plane[:, :, 0] = raw[::2, ::2]
        plane[:, :, 1] = raw[::2, 1::2]
        plane[:, :, 2] = raw[1::2, ::2]
        plane[:, :, 3] = raw[1::2, 1::2]

        if bayerformat is "rggb":
            R = plane[:, :, 0]
            Gr = plane[:, :, 1]
            Gb = plane[:, :, 2]
            B = plane[:, :, 3]
        elif bayerformat is "bggr":
            B = plane[:, :, 0]
            Gb = plane[:, :, 1]
            Gr = plane[:, :, 2]
            R = plane[:, :, 3]
        elif bayerformat is "gbrg":
            Gb = plane[:, :, 0]
            B = plane[:, :, 1]
            R = plane[:, :, 2]
            Gr = plane[:, :, 3]
        elif bayerformat is "grbg":
            Gr = plane[:, :, 0]
            R = plane[:, :, 1]
            B = plane[:, :, 2]
            Gb = plane[:, :, 3]

        raw_bayer = numpy.zeros((int(raw.shape[0] / 2), int(raw.shape[1] / 2), 4))
        raw_bayer[:, :, 0] = R
        raw_bayer[:, :, 1] = Gr
        raw_bayer[:, :, 2] = Gb
        raw_bayer[:, :, 3] = B

        return raw_bayer

    elif outputformat is "rgb":
        bilinear_interpolation(raw, bayerformat)

    return
