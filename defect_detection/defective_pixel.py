from preprocess import preprocess
import numpy as np


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
    roiSize = 15

    # 对四个边缘进行填补（延展长度roiSize / 2)
    ID_mirror = np.pad(ID, roiSize - 1, "symmetric")

    h_mirror = ID_mirror.shape[0]
    w_mirror = ID_mirror.shape[1]
    # 对奇偶行进行交换
    ID_mirror[[1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12], :] = ID_mirror[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], :]
    ID_mirror[[h_mirror-13, h_mirror-14, h_mirror-11, h_mirror-12, h_mirror-9, h_mirror-10, h_mirror-7, h_mirror-8, h_mirror-5, h_mirror-6, h_mirror-3, h_mirror-4, h_mirror-1, h_mirror-2], :] = ID_mirror[[h_mirror-14, h_mirror-13, h_mirror-12, h_mirror-11, h_mirror-10, h_mirror-9, h_mirror-8, h_mirror-7, h_mirror-6, h_mirror-5, h_mirror-4, h_mirror-3, h_mirror-2, h_mirror-1], :]

    # 对奇偶列进行交换
    ID_mirror[:, [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12]] = ID_mirror[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
    ID_mirror[:, [w_mirror-13, w_mirror-14, w_mirror-11, w_mirror-12, w_mirror-9, w_mirror-10, w_mirror-7, w_mirror-8, w_mirror-5, w_mirror-6, w_mirror-3, w_mirror-4, w_mirror-1, w_mirror-2]] = ID_mirror[:, [w_mirror-14, w_mirror-13, w_mirror-12, w_mirror-11, w_mirror-10, w_mirror-9, w_mirror-8, w_mirror-7, w_mirror-6, w_mirror-5, w_mirror-4, w_mirror-3, w_mirror-2, w_mirror-1]]

    print()
