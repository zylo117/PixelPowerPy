from read_bin.preprocess import preprocess
import numpy as np


def oc(IDraw, bayerformat="rggb", pedestal=64, bitdepth=10):
    IDyuv = preprocess(IDraw, outputformat="yuv")
    IDy = IDyuv[:, :, 0]
    (h, w) = IDy.shape
    h_center = h / 2 + 0.5
    w_center = w / 2 + 0.5

    roi_size = 100
    roi_size_half = roi_size / 2

    # 定义ROI中心区
    # [中，顶，右，底，左]
    roi_center_x = [w_center, w_center, w_center + h_center - (roi_size_half + 0.5), w_center, w_center - h_center + roi_size_half + 0.5]
    roi_center_y = [h_center, roi_size_half + 0.5, h_center, h - (roi_size_half - 0.5), h_center]

    threshold_data = np.zeros(len(roi_center_x))

    print()
