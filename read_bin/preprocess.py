import array
import numpy
from functionset import *


def preprocess(imageinput, bayerformat="rggb", outputformat="raw", mode=0, bitdepth=10, pedestal=64, FOV=0,
               whitebalance=True, signed=True, more_precise=False):
    # unsigned integer, 16位
    ID = array.array('H', open(imageinput, "rb").read())

    width = ID[0]
    height = ID[1]

    ID = numpy.array(ID).astype(numpy.double)
    ID = ID[2:]

    ID = ID.reshape((height, width))
    print(ID[0][0], ID[0][1], ID[1][0], ID[1][1])

    # 按mode切片
    ID = crop_by_mode(ID, mode)

    # 增益
    ID = ID + pedestal
    if not signed:
        ID[ID < 0] = 0
    print(ID[0][0], ID[0][1], ID[1][0], ID[1][1])

    # 白平衡
    if whitebalance:
        ID = white_balance(ID)
        ID[ID > 2 ** bitdepth - 1] = 2 ** bitdepth - 1  # 防过饱和

    # 去镜头阴影
    if FOV is not 0:
        ID = lens_shading_correction(ID, 75)
        ID[ID > 2 ** bitdepth - 1] = 2 ** bitdepth - 1  # 防过饱和
    print(ID[0][0], ID[0][1], ID[1][0], ID[1][1])

    # 图像格式转换
    if outputformat is "raw":
        # 源格式输出
        if not more_precise:
            ID = numpy.round(ID)
        return ID

    elif outputformat is "bayer":
        # 分4层输出，强制转换到RGrGbB
        plane = numpy.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2), 4))
        R = numpy.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2)))
        Gr = numpy.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2)))
        Gb = numpy.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2)))
        B = numpy.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2)))

        plane[:, :, 0] = ID[::2, ::2]
        plane[:, :, 1] = ID[::2, 1::2]
        plane[:, :, 2] = ID[1::2, ::2]
        plane[:, :, 3] = ID[1::2, 1::2]

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

        bayer = numpy.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2), 4))
        bayer[:, :, 0] = R
        bayer[:, :, 1] = Gr
        bayer[:, :, 2] = Gb
        bayer[:, :, 3] = B

        if not more_precise:
            bayer = numpy.round(bayer)
        return bayer

    elif outputformat is "rgb":
        rgb = bilinear_interpolation(ID, bayerformat)
        if not more_precise:
            rgb = numpy.round(rgb)
        return rgb

    elif outputformat is "yuv":
        rgb = bilinear_interpolation(ID, bayerformat)
        yuv = numpy.zeros(rgb.shape)
        yuv[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        yuv[:, :, 1] = 2 ** (bitdepth - 1) - 0.1687 * rgb[:, :, 0] - 0.3313 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2]
        yuv[:, :, 2] = 2 ** (bitdepth - 1) + 0.5 * rgb[:, :, 0] - 0.4187 * rgb[:, :, 1] - 0.0813 * rgb[:, :, 2]

        if not more_precise:
            yuv = numpy.round(yuv)
        return yuv

    return
