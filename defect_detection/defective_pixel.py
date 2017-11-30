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
    ID_mirror = np.pad(ID, roiSize-1,"symmetric")
    
