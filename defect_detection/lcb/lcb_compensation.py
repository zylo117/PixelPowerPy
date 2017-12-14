import numpy as np
from io_bin import write_bin


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
def find_compensation_area(lcb_image_data_bayer, threshold_r=4.8, threshold_gr=3.1, threshold_gb=3.1, threshold_b=4.8, mark_dust=False, offset=0):
    lcb_image_data_r = lcb_image_data_bayer[:, :, 0]
    lcb_image_data_gr = lcb_image_data_bayer[:, :, 1]
    lcb_image_data_gb = lcb_image_data_bayer[:, :, 2]
    lcb_image_data_b = lcb_image_data_bayer[:, :, 3]

    if not mark_dust:
        lcb_r = np.where(lcb_image_data_r > threshold_r)
        lcb_gr = np.where(lcb_image_data_gr > threshold_gr)
        lcb_gb = np.where(lcb_image_data_gb > threshold_gb)
        lcb_b = np.where(lcb_image_data_b > threshold_b)
    else:
        lcb_r = np.where(lcb_image_data_r > threshold_r)
        lcb_r = [np.min(lcb_r[0]) - offset, np.max(lcb_r[0]) + offset, np.min(lcb_r[1]) - offset, np.max(lcb_r[1]) + offset]
        lcb_gr = np.where(lcb_image_data_gr > threshold_gr)
        lcb_gr = [np.min(lcb_gr[0]) - offset, np.max(lcb_gr[0]) + offset, np.min(lcb_gr[1]) - offset, np.max(lcb_gr[1]) + offset]
        lcb_gb = np.where(lcb_image_data_gb > threshold_gb)
        lcb_gb = [np.min(lcb_gb[0]) - offset, np.max(lcb_gb[0]) + offset, np.min(lcb_gb[1]) - offset, np.max(lcb_gb[1]) + offset]
        lcb_b = np.where(lcb_image_data_b > threshold_b)
        lcb_b = [np.min(lcb_b[0]) - offset, np.max(lcb_b[0]) + offset, np.min(lcb_b[1]) - offset, np.max(lcb_b[1]) + offset]

    return lcb_r, lcb_gr, lcb_gb, lcb_b


# 把每个颜色的暗区（故障区）对应的单色RAW图的坐标簇进行增益，然后再4色合成bayer图
def lcb_brightness_compensation(raw_image_data, lcb_image_data_bayer, threshold_r=4.8, threshold_gr=3.1, threshold_gb=3.1,
                                threshold_b=4.8,
                                bin_padR=0, bin_padC=4, roiSize=[13, 13], fix_dust=False, dust_offset=3):
    raw_loc_x, raw_loc_y = lcb_coordinate(lcb_image_data_bayer, bin_padR, bin_padC, roiSize)

    lcb_r, lcb_gr, lcb_gb, lcb_b = find_compensation_area(lcb_image_data_bayer, threshold_r, threshold_gr, threshold_gb,
                                                          threshold_b, mark_dust=fix_dust, offset=dust_offset)

    lcb_image_height, lcb_image_width, channel = lcb_image_data_bayer.shape

    if not fix_dust:
        for i in range(len(lcb_r[0])):
            if 0 < lcb_r[0][i] < lcb_image_height - 1 and 0 < lcb_r[1][i] < lcb_image_width - 1:
                lcb_gain(raw_image_data[:, :, 0],
                         raw_loc_x[lcb_r[0][i]],
                         raw_loc_y[lcb_r[1][i]],
                         lcb_image_data_bayer[lcb_r[0][i], lcb_r[1][i], 0])

                lcb_image_data_bayer[lcb_r[0][i], lcb_r[1][i], 0] = 0

        for i in range(len(lcb_gr[0])):
            if 0 < lcb_gr[0][i] < lcb_image_height - 1 and 0 < lcb_gr[1][i] < lcb_image_width - 1:
                lcb_gain(raw_image_data[:, :, 1],
                         raw_loc_x[lcb_gr[0][i]],
                         raw_loc_y[lcb_gr[1][i]],
                         lcb_image_data_bayer[lcb_gr[0][i], lcb_gr[1][i], 1])

                lcb_image_data_bayer[lcb_gr[0][i], lcb_gr[1][i], 1] = 0

        for i in range(len(lcb_gb[0])):
            if 0 < lcb_gb[0][i] < lcb_image_height - 1 and 0 < lcb_gb[1][i] < lcb_image_width - 1:
                lcb_gain(raw_image_data[:, :, 2],
                         raw_loc_x[lcb_gb[0][i]],
                         raw_loc_y[lcb_gb[1][i]],
                         lcb_image_data_bayer[lcb_gb[0][i], lcb_gb[1][i], 2])

                lcb_image_data_bayer[lcb_gb[0][i], lcb_gb[1][i], 2] = 0

        for i in range(len(lcb_b[0])):
            if 0 < lcb_b[0][i] < lcb_image_height - 1 and 0 < lcb_b[1][i] < lcb_image_width - 1:
                lcb_gain(raw_image_data[:, :, 3],
                         raw_loc_x[lcb_b[0][i]],
                         raw_loc_y[lcb_b[1][i]],
                         lcb_image_data_bayer[lcb_b[0][i], lcb_b[1][i], 3])

                lcb_image_data_bayer[lcb_b[0][i], lcb_b[1][i], 3] = 0

    else:
        lcb_centrosymmetric_patching(lcb_image_data_bayer, lcb_r, lcb_gr, lcb_gb, lcb_b)

    write_bin.array2bin(raw_image_data, inputformat="bayer")

    return lcb_image_data_bayer


# 对应的单色RAW图的坐标簇进行增益
def lcb_gain(raw_image_data_single_color, y, x, lcb_val, roiSize=[13, 13], calculation="simple"):
    if calculation is "simple":
        raw_image_data_single_color[y - roiSize[1] // 2:y + roiSize[1] // 2,
                                    x - roiSize[0] // 2:x + roiSize[0] // 2] + lcb_val

    elif calculation is "centrosymmetric":
        height, width = raw_image_data_single_color.shape[:2]

        raw_image_data_single_color[y - roiSize[1]**2 // 2:y + roiSize[1]**2 // 2,
                                    x - roiSize[0]**2 // 2:x + roiSize[0]**2 // 2] = \
        raw_image_data_single_color[height - 1 - (y + roiSize[1]**2 // 2):height - 1 - (y - roiSize[1]**2 // 2),
                                    width - 1 - (x + roiSize[0]**2 // 2):width - 1 - (x - roiSize[0]**2 // 2)]


# 对应的单色RAW图的坐标簇进行中心对称复制修补
def lcb_centrosymmetric_patching(raw_image_data_bayer, lcb_r, lcb_gr, lcb_gb, lcb_b):

    height, width, c = raw_image_data_bayer.shape

    lcb_data = np.vstack((lcb_r, lcb_gr, lcb_gb, lcb_b))

    for z in range(c):
        raw_image_data_bayer[lcb_data[z][0]: lcb_data[z][1], lcb_data[z][2]:lcb_data[z][3], z] = \
        raw_image_data_bayer[height - 1 - lcb_data[z][1]: height - 1 - lcb_data[z][0], width - 1 - lcb_data[z][3]:width - 1 -lcb_data[z][2], z]
