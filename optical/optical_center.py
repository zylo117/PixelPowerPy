from read_bin.preprocess import preprocess
import numpy as np


def oc(IDraw, bayerformat="rggb", pedestal=64, bitdepth=10):
    IDyuv = preprocess(IDraw, outputformat="yuv")
    IDy = IDyuv[:, :, 0]
    (h, w) = IDy.shape
    h_center = h / 2 + 0.5 - 1
    w_center = w / 2 + 0.5 - 1

    roi_size = 100
    roi_size_half = roi_size / 2

    # 定义ROI中心区
    # [中，顶，右，底，左]
    roi_center_x = [w_center, w_center, w_center + h_center - (roi_size_half + 0.5) + 1, w_center,
                    w_center - h_center + roi_size_half + 0.5 - 1]
    roi_center_y = [h_center, roi_size_half + 0.5 - 1, h_center, h - (roi_size_half - 0.5) - 1, h_center]

    threshold_data = np.zeros(len(roi_center_x))

    for i in range(len(threshold_data)):
        x1 = int(roi_center_x[i] - (roi_size_half - 0.5))
        x2 = int(roi_center_x[i] + (roi_size_half - 0.5))
        y1 = int(roi_center_y[i] - (roi_size_half - 0.5))
        y2 = int(roi_center_y[i] + (roi_size_half - 0.5))
        threshold_data[i] = np.mean(IDy[y1:y2 + 1, x1:x2 + 1])
    print()
