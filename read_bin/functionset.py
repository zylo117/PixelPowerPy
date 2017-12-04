import numpy as np
import numba
from read_bin.conv2d_matlab import *


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


def white_balance(raw):
    width = raw.shape[1]
    height = raw.shape[0]

    # 分4层
    plane = np.zeros((int(height / 2), int(width / 2), 4))
    plane[:, :, 0] = raw[::2, ::2]
    plane[:, :, 1] = raw[::2, 1::2]
    plane[:, :, 2] = raw[1::2, ::2]
    plane[:, :, 3] = raw[1::2, 1::2]

    # print(plane[0][0][0], plane[0][1][0], plane[1][0][0], plane[1][1][0])
    # print(plane[0][0][1], plane[0][1][1], plane[1][0][1], plane[1][1][1])
    # print(plane[0][0][2], plane[0][1][2], plane[1][0][2], plane[1][1][2])
    # print(plane[0][0][3], plane[0][1][3], plane[1][0][3], plane[1][1][3])

    block_size_R = 100
    block_size_C = 100
    center = [plane.shape[0] / 2 - 1, plane.shape[1] / 2 - 1]

    center_block = plane[int(center[0] - block_size_R / 2 + 1):int(center[0] + int(block_size_R / 2 + 1)),
                   int(center[1] - int(block_size_C / 2) + 1): int(center[1] + int(block_size_C / 2) + 1), :]

    center_mean = np.mean((np.mean(center_block, axis=0)), axis=0)
    balance = np.max(center_mean) / center_mean

    for i in range(4):
        plane[:, :, i] = plane[:, :, i] * balance[i]

    # print(plane[0][0][0], plane[0][1][0], plane[1][0][0], plane[1][1][0])
    # print(plane[0][0][1], plane[0][1][1], plane[1][0][1], plane[1][1][1])
    # print(plane[0][0][2], plane[0][1][2], plane[1][0][2], plane[1][1][2])
    # print(plane[0][0][3], plane[0][1][3], plane[1][0][3], plane[1][1][3])

    # 4合1
    post_wb = np.zeros(raw.shape)
    post_wb[::2, ::2] = plane[:, :, 0]
    post_wb[::2, 1::2] = plane[:, :, 1]
    post_wb[1::2, ::2] = plane[:, :, 2]
    post_wb[1::2, 1::2] = plane[:, :, 3]
    # print(post_wb[0][0], post_wb[0][1], post_wb[1][0], post_wb[1][1])

    return post_wb


# 镜头阴影纠正
# 涉及遍历全像素，使用Numba辅助加速
@numba.jit()
def lens_shading_correction(raw, FOV):
    width = raw.shape[1]
    height = raw.shape[0]

    # 找出中心并计算外接圆半径。加0.5的原因：一般来说，真实的像素是方形的，而真正的图像中心是四个像素中间的间隙，而不是某个像素
    centerX = width / 2 + 0.5 - 1
    centerY = height / 2 + 0.5 - 1
    circumradius = (centerX ** 2 + centerY ** 2) ** 0.5

    # 求出所有点所在的视野（半）角对应程度
    for j in range(height):
        for i in range(width):
            FOV_scale = (FOV / 2) * ((centerX - i) ** 2 + (centerY - j) ** 2) ** 0.5 / circumradius

            # 采用4次余弦因子增益，次方越大，纠正效果越强(具体函数，是根据图像对角线分布函数拟合的)
            lsc_factor = 1 / (np.cos(np.pi / 180 * FOV_scale)) ** 4
            raw[j, i] = raw[j, i] * lsc_factor

    return raw


# 线性插值
def bilinear_interpolation(raw, bayerformat):
    w = raw.shape[1]
    h = raw.shape[0]
    if bayerformat is "rggb":
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
    R = raw * red_mask
    G = raw * green_mask
    B = raw * blue_mask

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
