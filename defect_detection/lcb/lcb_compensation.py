import numpy as np


# 找出LCB图对应的RAW原图的坐标簇中心，到时候整簇进行调整
def lcb_coordinate(lcb_image_data_bayer, bin_padR=0, bin_padC=4, roiSize=[13, 13]):
    scaled_height, scaled_width, c = lcb_image_data_bayer.shape

    raw_loc_x = np.zeros((scaled_width, c))
    raw_loc_y = np.zeros((scaled_height, c))

    for z in range(c):
        raw_loc_x = np.arange(roiSize[0] // 2, scaled_width * roiSize[0] - roiSize[0] // 2, roiSize[0]) + bin_padC
        raw_loc_y = np.arange(roiSize[1] // 2, scaled_height * roiSize[1] - roiSize[1] // 2, roiSize[1]) + bin_padR

    return raw_loc_x, raw_loc_y


# 找出超出阀值的暗区（故障区）
def find_compensation_area(lcb_image_data_bayer, threshold_r=4.8, threshold_gr=3.1, threshold_gb=3.1, threshold_b=4.8):
    lcb_image_data_r = lcb_image_data_bayer[:, :, 0]
    lcb_image_data_gr = lcb_image_data_bayer[:, :, 1]
    lcb_image_data_gb = lcb_image_data_bayer[:, :, 2]
    lcb_image_data_b = lcb_image_data_bayer[:, :, 3]

    lcb_r = np.where(lcb_image_data_r > threshold_r)
    lcb_gr = np.where(lcb_image_data_gr > threshold_gr)
    lcb_gb = np.where(lcb_image_data_gb > threshold_gb)
    lcb_b = np.where(lcb_image_data_b > threshold_b)

    return lcb_r, lcb_gr, lcb_gb, lcb_b


# 把每个颜色的暗区（故障区）对应的单色RAW图的坐标簇进行增益，然后再4色合成bayer图
def lcb_brightness_compensation(lcb_image_data_bayer, threshold_r=4.8, threshold_gr=3.1, threshold_gb=3.1,
                                threshold_b=4.8,
                                bin_padR=0, bin_padC=4, roiSize=[13, 13]):
    raw_loc_x, raw_loc_y = lcb_coordinate(lcb_image_data_bayer, bin_padR, bin_padC, roiSize)

    lcb_r, lcb_gr, lcb_gb, lcb_b = find_compensation_area(lcb_image_data_bayer, threshold_r, threshold_gr, threshold_gb,
                                                          threshold_b)

    lcb_image_height, lcb_image_width, channel = lcb_image_data_bayer.shape

    r_cluster_set = []
    gr_cluster_set = []
    gb_cluster_set = []
    b_cluster_set = []

    for i in range(len(lcb_r[0])):
        if 0 < lcb_r[0][i] < lcb_image_height - 1 and 0 < lcb_r[1][i] < lcb_image_width - 1:
            r_cluster_set.append(
                (raw_loc_x[lcb_r[0][i]], raw_loc_y[lcb_r[1][i]], lcb_image_data_bayer[lcb_r[0][i], lcb_r[1][i], 0]))

    for i in range(len(lcb_gr[0])):
        if 0 < lcb_gr[0][i] < lcb_image_height - 1 and 0 < lcb_gr[1][i] < lcb_image_width - 1:
            gr_cluster_set.append(
                (raw_loc_x[lcb_gr[0][i]], raw_loc_y[lcb_gr[1][i]], lcb_image_data_bayer[lcb_gr[0][i], lcb_gr[1][i], 1]))

    for i in range(len(lcb_gb[0])):
        if 0 < lcb_gb[0][i] < lcb_image_height - 1 and 0 < lcb_gb[1][i] < lcb_image_width - 1:
            gb_cluster_set.append(
                (raw_loc_x[lcb_gb[0][i]], raw_loc_y[lcb_gb[1][i]], lcb_image_data_bayer[lcb_gb[0][i], lcb_gb[1][i], 2]))

    for i in range(len(lcb_b[0])):
        if 0 < lcb_b[0][i] < lcb_image_height - 1 and 0 < lcb_b[1][i] < lcb_image_width - 1:
            b_cluster_set.append(
                (raw_loc_x[lcb_b[0][i]], raw_loc_y[lcb_b[1][i]], lcb_image_data_bayer[lcb_b[0][i], lcb_b[1][i], 3]))

    return 0


# 对应的单色RAW图的坐标簇进行增益
def lcb_gain(raw_image_data_single_color, y, x, lcb_val, roiSize=[13, 13]):
    raw_image_data_single_color[y - roiSize[1] // 2:y + roiSize[1] // 2,
                                x - roiSize[0] // 2:x + roiSize[0] // 2] - lcb_val
