from read_bin.preprocess import preprocess
from read_bin.conv2d_matlab import conv2
import numpy as np
from scipy.ndimage.filters import correlate
import numba


def dp(raw, bayerformat="rggb", pedestal=64, bitdepth=10, threshold_defect=0.19, threshold_defectLow=0.12,
       cluster_type="bayer", cluster_size=3, neighbour_type="avg", more_precise=False):
    if threshold_defect > 1:
        ID = preprocess(raw, bayerformat, outputformat="raw", mode=0, bitdepth=10, pedestal=64, FOV=0,
                        whitebalance=False, signed=True, more_precise=more_precise)
    else:
        ID = preprocess(raw, bayerformat, outputformat="raw", mode=0, bitdepth=10, pedestal=64, FOV=0,
                        whitebalance=True, signed=True, more_precise=more_precise)

    h = ID.shape[0]
    w = ID.shape[1]

    # 初始化变量
    count_cluster = 0
    count_DP = 0
    count_NDP = 0
    count_NDPP = 0
    roi_size = 15

    # 对四个边缘进行填补（延展长度roiSize / 2)
    ID_mirror = np.pad(ID, roi_size - 1, "symmetric")

    h_mirror = ID_mirror.shape[0]
    w_mirror = ID_mirror.shape[1]

    # 对奇偶行进行交换
    ID_mirror[[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12], :] = ID_mirror[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], :]
    ID_mirror[[h_mirror-13, h_mirror-14, h_mirror-11, h_mirror-12, h_mirror-9, h_mirror-10, h_mirror-7, h_mirror-8, h_mirror-5, h_mirror-6, h_mirror-3, h_mirror-4, h_mirror-1, h_mirror-2], :] = ID_mirror[[h_mirror-14, h_mirror-13, h_mirror-12, h_mirror-11, h_mirror-10, h_mirror-9, h_mirror-8, h_mirror-7, h_mirror-6, h_mirror-5, h_mirror-4, h_mirror-3, h_mirror-2, h_mirror-1], :]
    # 对奇偶列进行交换
    ID_mirror[:, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]] = ID_mirror[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
    ID_mirror[:, [w_mirror-13, w_mirror-14, w_mirror-11, w_mirror-12, w_mirror-9, w_mirror-10, w_mirror-7, w_mirror-8, w_mirror-5, w_mirror-6, w_mirror-3, w_mirror-4, w_mirror-1, w_mirror-2]] = ID_mirror[:, [w_mirror-14, w_mirror-13, w_mirror-12, w_mirror-11, w_mirror-10, w_mirror-9, w_mirror-8, w_mirror-7, w_mirror-6, w_mirror-5, w_mirror-4, w_mirror-3, w_mirror-2, w_mirror-1]]

    # 把图像进行均值归一化
    ID_avg = np.zeros(ID_mirror.shape)
    kernel = np.zeros((2 * roi_size - 1, 2 * roi_size - 1))
    kernel[::2, ::2] = 1 / (roi_size ** 2)
    ID_avg = correlate(ID_mirror, kernel)
    # 去除多余边框，恢复原分辨率
    ID_avg = ID_avg[roi_size - 1: h_mirror - roi_size + 1, roi_size - 1: w_mirror - roi_size + 1]

    # 找出defective pixels
    if threshold_defect > 1:
        ID_delta = ID - ID_avg  # LSB（Least significant bit） 值，最小有效值，黑场（dark field）测试条件
        map_defect = (np.abs(ID_delta) > threshold_defect).astype(np.double)  # 求出大于threshold的布尔坐标图，转换为double（0或1）
    else:
        # 计算差异百分比
        ID_percDiff = np.abs((ID - ID_avg) / ID_avg)
        ID_percDiff[np.isnan(ID_percDiff)] = 0  # 把所有0/0的无效值替换为0
        ID_delta = ID_percDiff
        map_defect = (np.abs(ID_delta) > threshold_defect).astype(np.double)  # 求出大于threshold的布尔坐标图，转换为double（0或1）
        ID_delta = ID_delta * 100  # 小数转百分比

    # 定义kernel

    # 1.簇（cluster）内核
    # 当内核对应点的值，大于100 + cluster_size - 1的时候，该点就被标记为cluster的中心。后期增长（grow）cluster的时候，内核会被再次应用
    if cluster_type is "bayer":
        cluster_pattern = np.array([[1, 0, 1, 0, 1],
                                   [0, 0, 0, 0, 0],
                                   [1, 0, 100, 0, 1],
                                   [0, 0, 0, 0, 0],
                                   [1, 0, 1, 0, 1]])
        
    elif cluster_type is "raw":
        cluster_pattern = np.array([[1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 100, 1, 1],
                                   [1, 1, 1, 1, 1],
                                   [1, 1, 1, 1, 1]])

    # 2.梯对（ladder pair）内核
    ladder_pattern = np.array([[0, 1, 0, 1, 0],
                              [0, 0, 0, 0, 0],
                              [0, 1, 33, 1, 0],
                              [0, 0, 0, 0, 0],
                              [0, 1, 0, 1, 0]])

    # 3.对（pair）内核
    pair_pattern = np.array([[1, 0, 1, 0, 1],
                            [0, 0, 0, 0, 0],
                            [1, 0, 33, 0, 1],
                            [0, 0, 0, 0, 0],
                            [1, 0, 1, 0, 1]])

    # 4.行（row）内核
    row_pattern = np.array([[0, 0, 0, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 33, 0, 0],
                           [1, 1, 1, 1, 1],
                           [0, 0, 0, 0, 0]])

    # 1.簇（cluster）检测
    # 检测cluster defects，并标记cluster里面的所有像素
    map_temp_cluster = conv2(map_defect, cluster_pattern)  # 应用卷积
    map_temp_cluster = (np.abs(map_temp_cluster) >= 100 + cluster_size - 1).astype(np.double)
    map_temp_cluster = conv2(map_defect, cluster_pattern)  # 再次应用卷积，找出影响区
    map_temp_cluster = map_temp_cluster * map_defect  # 标出影响区里面的所有像素
    map_temp_cluster = (map_temp_cluster > 0).astype(np.double)
    print()
