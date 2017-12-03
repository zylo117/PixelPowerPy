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

    # 计算并应用二值化（threshold）
    oc_threshold = np.mean([threshold_data[0], np.mean(threshold_data[1:])])  # 求出二值化的阀值
    ID_threshold_binary = (IDy >= oc_threshold).astype(np.double)  # 浮点布尔图，False为暗，True为亮

    # 计算重心
    total_points = int(np.sum(ID_threshold_binary))
    row_weight = np.arange(1, h + 1)
    col_weight = np.arange(1, w + 1)
    row_sum = np.sum(ID_threshold_binary, axis=1) * row_weight
    col_sum = np.sum(ID_threshold_binary, axis=0) * col_weight

    oc_x = np.sum(col_sum) / total_points
    oc_y = np.sum(row_sum) / total_points
    oc_x_shift = oc_x - (IDy.shape[1] / 2 + 0.5)
    oc_y_shift = (IDy.shape[0] / 2 + 0.5) - oc_y

    # 半径
    oc_mag_shift = np.sqrt(oc_x_shift**2 + oc_y_shift**2)

    output = [oc_threshold, oc_x, oc_y, oc_x_shift, oc_y_shift, oc_mag_shift]

    return output
