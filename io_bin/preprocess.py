import cv2
import numpy as np
from math_tool import conv2
from read_bin import BinFile


def preprocess(imageinput, bayerformat="rggb", outputformat="raw", mode=0, bitdepth=10, pedestal=64, FOV=0,
               whitebalance=True, signed=True, more_precise=False, custom_size=[0, 0], custom_encoding="H", for_SFR_test=False):

    if custom_encoding is "H":
        # 默认unsigned integer, 16位
        bin = BinFile(imageinput, datatype=np.uint16)
        header, ID = bin.get_realdata(2)
    else:
        bin = BinFile(imageinput, datatype=custom_encoding)
        header, ID = bin.get_realdata(2)

    if custom_size == [0, 0]:
        width = header[0]
        height = header[1]
    else:
        width = custom_size[0]
        height = custom_size[1]

    ID = np.asarray(ID).astype(np.double)

    if custom_size == [0, 0]:
        ID = ID[2:]

    ID = ID.reshape((height, width))

    # 按mode切片
    ID = crop_by_mode(ID, mode)

    # 增益
    ID = ID + pedestal
    if not signed:
        ID[ID < 0] = 0

    # 白平衡
    if whitebalance:
        ID = white_balance(ID, for_SFR_test)
        ID[ID > 2 ** bitdepth - 1] = 2 ** bitdepth - 1  # 防过饱和

    # 去镜头阴影
    if FOV is not 0:
        ID = lens_shading_correction(ID, 75, more_precise=more_precise)
        ID[ID > 2 ** bitdepth - 1] = 2 ** bitdepth - 1  # 防过饱和

    # 图像格式转换
    if outputformat is "raw":
        # 源格式输出
        if not more_precise:
            ID = np.round(ID)
        return ID

    elif outputformat is "bayer":
        # 分4层输出，强制转换到RGrGbB
        plane = np.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2), 4))
        R = np.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2)))
        Gr = np.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2)))
        Gb = np.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2)))
        B = np.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2)))

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

        bayer = np.zeros((int(ID.shape[0] / 2), int(ID.shape[1] / 2), 4))
        bayer[:, :, 0] = R
        bayer[:, :, 1] = Gr
        bayer[:, :, 2] = Gb
        bayer[:, :, 3] = B

        if not more_precise:
            bayer = np.round(bayer)
        return bayer

    elif outputformat is "rgb":
        bgr = bilinear_interpolation_opencv(ID, bayerformat)
        if not more_precise:
            bgr = np.round(bgr)
        return bgr

    elif outputformat is "yuv":
        bgr = bilinear_interpolation(ID, bayerformat)

        # opencv solution
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        # numpy solution
        # yuv = np.zeros(rgb.shape)
        # yuv[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        # yuv[:, :, 1] = 2 ** (bitdepth - 1) - 0.1687 * rgb[:, :, 0] - 0.3313 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2]
        # yuv[:, :, 2] = 2 ** (bitdepth - 1) + 0.5 * rgb[:, :, 0] - 0.4187 * rgb[:, :, 1] - 0.0813 * rgb[:, :, 2]

        if not more_precise:
            yuv = np.round(yuv)
        return yuv

    return


def crop_by_mode(raw, mode):
    width = raw.shape[1]
    height = raw.shape[0]

    # 按mode切片
    if mode is 1:
        raw = raw[8:height - 8, 8:width - 8]
    elif mode is 2:
        raw = raw[int((height - (width - 16) / (16 / 9)) / 2): height - int((height - (width - 16) / (16 / 9)) / 2),
              8:width - 8]
    elif mode is 3:
        raw = raw[8:height - 8,
              int((width - (height - 16) * (3 / 2)) / 2): width - int((width - (height - 16) * (3 / 2)) / 2)]
    elif mode is 4:
        raw = raw[8:height - 8,
              int((width - (height - 16) * (4 / 3)) / 2): width - int((width - (height - 16) * (4 / 3)) / 2)]

    return raw


def white_balance(raw, for_SFR_test=False):
    width = raw.shape[1]
    height = raw.shape[0]

    # 分4层
    plane = np.zeros((int(height / 2), int(width / 2), 4))
    plane[:, :, 0] = raw[::2, ::2]
    plane[:, :, 1] = raw[::2, 1::2]
    plane[:, :, 2] = raw[1::2, ::2]
    plane[:, :, 3] = raw[1::2, 1::2]

    block_size_R = 100
    block_size_C = 100

    if not for_SFR_test:
        center = [plane.shape[0] / 2 - 1, plane.shape[1] / 2 - 1]
    else:
        center = [plane.shape[0] / 2 - 1, plane.shape[1] / 2 - 1 - 0.08 * plane.shape[1]]  # slightly dodge the dot on the very center of the SFR chart

    center_block = plane[int(center[0] - block_size_R / 2 + 1):int(center[0] + int(block_size_R / 2 + 1)),
                   int(center[1] - int(block_size_C / 2) + 1): int(center[1] + int(block_size_C / 2) + 1), :]

    center_mean = np.mean((np.mean(center_block, axis=0)), axis=0)
    balance = np.max(center_mean) / center_mean

    for i in range(4):
        plane[:, :, i] = plane[:, :, i] * balance[i]

    # 4合1
    post_wb = np.zeros(raw.shape)
    post_wb[::2, ::2] = plane[:, :, 0]
    post_wb[::2, 1::2] = plane[:, :, 1]
    post_wb[1::2, ::2] = plane[:, :, 2]
    post_wb[1::2, 1::2] = plane[:, :, 3]

    return post_wb


# 镜头阴影纠正
# 涉及遍历全像素，使用Numba加速
# @numba.jit()
def lens_shading_correction(raw, FOV):
    width = raw.shape[1]
    height = raw.shape[0]

    # 找出中心并计算外接圆半径。加0.5的原因：一般来说，真实的像素是方形的，而真正的图像中心是四个像素中间的间隙，而不是某个像素
    centerX = width / 2 + 0.5 - 1
    centerY = height / 2 + 0.5 - 1
    circumradius = (centerX ** 2 + centerY ** 2) ** 0.5

    # 求出所有点所在的视野（半）角对应程度
    # pure python, slow "for loop"
    # Old Method, relied on Numba to accelerate
    # for j in range(height):
    #     for i in range(width):
    #         if not more_precise:
    #             FOV_scale = (FOV / 2) * ((centerX - i) ** 2 + (centerY - j) ** 2) ** 0.5 / circumradius
    #         else:
    #             FOV_scale = 180 / np.pi * np.arctan(np.tan(np.pi / 180 * (FOV / 2)) * ((centerX - i) ** 2 + (centerY - j) ** 2) ** 0.5 / circumradius)
    #
    #         # 采用4次余弦因子增益，次方越大，纠正效果越强(具体函数，是根据图像对角线分布函数拟合的)
    #         lsc_factor = 1 / (np.cos(np.pi / 180 * FOV_scale)) ** 4
    #         raw[j, i] = raw[j, i] * lsc_factor

    # New Method, no more "for loop", don't relied on Numba
    x_points = (centerX - np.arange(0, width)) ** 2
    y_points = (centerY - np.arange(0, height)) ** 2

    x_grid = np.tile(x_points, [len(y_points), 1])
    y_grid = np.transpose(np.tile(y_points, [len(x_points), 1]))

    FOV_grid = 180 / np.pi * np.arctan(np.tan(np.pi / 180 * (FOV / 2)) * np.sqrt(x_grid + y_grid) / circumradius)
    lsc_grid = 1 / np.cos(np.pi / 180 * FOV_grid) ** 4

    # Apply correction coefficient
    raw = lsc_grid * raw

    return raw


# 线性插值
def bilinear_interpolation_opencv(raw_bayer, bayerformat="rggb"):
    if str.lower(bayerformat) is "rggb":
        raw_bayer = cv2.cvtColor(raw_bayer, cv2.COLOR_BAYER_BG2BGR)
    elif str.lower(bayerformat) is "bggr":
        raw_bayer = cv2.cvtColor(raw_bayer, cv2.COLOR_BAYER_RG2BGR)
    elif str.lower(bayerformat) is "gbrg":
        raw_bayer = cv2.cvtColor(raw_bayer, cv2.COLOR_BAYER_GR2BGR)
    elif str.lower(bayerformat) is "grbg":
        raw_bayer = cv2.cvtColor(raw_bayer, cv2.COLOR_BAYER_GB2BGR)


def bilinear_interpolation(raw_bayer, bayerformat="rggb"):
    w = raw_bayer.shape[1]
    h = raw_bayer.shape[0]
    if str.lower(bayerformat) is "rggb":
        red_mask = np.tile(([1, 0], [0, 0]), [int(h / 2), int(w / 2)])
        green_mask = np.tile(([0, 1], [1, 0]), [int(h / 2), int(w / 2)])
        blue_mask = np.tile(([0, 0], [0, 1]), [int(h / 2), int(w / 2)])
    elif str.lower(bayerformat) is "bggr":
        blue_mask = np.tile(([1, 0], [0, 0]), [int(h / 2), int(w / 2)])
        green_mask = np.tile(([0, 1], [1, 0]), [int(h / 2), int(w / 2)])
        red_mask = np.tile(([0, 0], [0, 1]), [int(h / 2), int(w / 2)])
    elif str.lower(bayerformat) is "gbrg":
        green_mask = np.tile(([1, 0], [0, 0]), [int(h / 2), int(w / 2)])
        blue_mask = np.tile(([0, 1], [1, 0]), [int(h / 2), int(w / 2)])
        red_mask = np.tile(([0, 0], [0, 1]), [int(h / 2), int(w / 2)])
    elif str.lower(bayerformat) is "grbg":
        green_mask = np.tile(([1, 0], [0, 0]), [int(h / 2), int(w / 2)])
        red_mask = np.tile(([0, 1], [1, 0]), [int(h / 2), int(w / 2)])
        blue_mask = np.tile(([0, 0], [0, 1]), [int(h / 2), int(w / 2)])

    # 分离RG（Gr + Gb）B
    R = raw_bayer * red_mask
    G = raw_bayer * green_mask
    B = raw_bayer * blue_mask

    # 给缺失区插值
    R = R + conv2(R, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="same")
    G = G + conv2(G, np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]), mode="same")
    B = B + conv2(B, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]), mode="same")

    # 计算每个像素由多少个像素合成
    Rpix = conv2(red_mask, np.ones((3, 3)), "same")
    Gpix = conv2(green_mask, np.ones((3, 3)), "same")
    Gpix = Gpix - green_mask * Gpix + green_mask
    Bpix = conv2(blue_mask, np.ones((3, 3)), "same")

    R = R / Rpix
    G = G / Gpix
    B = B / Bpix

    rgb = np.zeros((h, w, 3))
    rgb[:, :, 0] = R
    rgb[:, :, 1] = G
    rgb[:, :, 2] = B

    return rgb
