import array
import numpy as np
import numba
from read_bin.conv2d_matlab import conv2


def preprocess(imageinput, bayerformat="rggb", outputformat="raw", mode=0, bitdepth=10, pedestal=64, FOV=0,
               whitebalance=True, signed=True, more_precise=False):
    # unsigned integer, 16位
    ID = array.array('H', open(imageinput, "rb").read())

    width = ID[0]
    height = ID[1]

    ID = np.array(ID).astype(np.double)
    ID = ID[2:]

    ID = ID.reshape((height, width))
    # print(ID[0][0], ID[0][1], ID[1][0], ID[1][1])

    # 按mode切片
    ID = crop_by_mode(ID, mode)

    # 增益
    ID = ID + pedestal
    if not signed:
        ID[ID < 0] = 0
    # print(ID[0][0], ID[0][1], ID[1][0], ID[1][1])

    # 白平衡
    if whitebalance:
        ID = white_balance(ID)
        ID[ID > 2 ** bitdepth - 1] = 2 ** bitdepth - 1  # 防过饱和

    # 去镜头阴影
    if FOV is not 0:
        ID = lens_shading_correction(ID, 75)
        ID[ID > 2 ** bitdepth - 1] = 2 ** bitdepth - 1  # 防过饱和
    # print(ID[0][0], ID[0][1], ID[1][0], ID[1][1])

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
        rgb = bilinear_interpolation(ID, bayerformat)
        if not more_precise:
            rgb = np.round(rgb)
        return rgb

    elif outputformat is "yuv":
        rgb = bilinear_interpolation(ID, bayerformat)
        yuv = np.zeros(rgb.shape)
        yuv[:, :, 0] = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        yuv[:, :, 1] = 2 ** (bitdepth - 1) - 0.1687 * rgb[:, :, 0] - 0.3313 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2]
        yuv[:, :, 2] = 2 ** (bitdepth - 1) + 0.5 * rgb[:, :, 0] - 0.4187 * rgb[:, :, 1] - 0.0813 * rgb[:, :, 2]

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
