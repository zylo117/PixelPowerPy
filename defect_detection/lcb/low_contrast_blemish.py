import numpy as np
import scipy.signal as spsignal

from matlab_tool import imfilter_with_1d_kernel, rescale_intensity
from preprocess import preprocess
from preprocess import bilinear_interpolation
from lcb import lcb_compensation


def lcb(IDraw, bayerformat="rggb", pedestal=64, bitdepth=10, mode=2, roiSize=[13, 13], filterWidth=9, threshold=12.6,
        interpolation=True, exceed2maxval=True, compensation=False):
    IDbayer = preprocess(IDraw, outputformat="bayer", mode=mode, more_precise=True)

    height = IDbayer.shape[0] * 2
    width = IDbayer.shape[1] * 2

    roiX_size = roiSize[1]
    roiY_size = roiSize[0]
    fw = filterWidth

    CornerROISize = [filterWidth + 1, filterWidth + 1]

    cornerX_size = CornerROISize[1]
    cornerY_size = CornerROISize[0]

    # define the filter
    # 可以看出，滤波后大于0的就是暗区（也就是说中心点比两边的像素值低）
    h = np.hstack((1 / 2, np.zeros((fw - 3) // 2), -1, np.zeros((fw - 3) // 2), 1 / 2))

    # scale down input image

    IDbin_all, bin_padR, bin_padC = binning(IDbayer);
    [rows, cols, c] = IDbin_all.shape

    # define final result
    I_filtered_raw = np.zeros((rows * 2, cols * 2))
    I_filtered_r = np.zeros((rows, cols))
    I_filtered_gr = np.zeros((rows, cols))
    I_filtered_gb = np.zeros((rows, cols))
    I_filtered_b = np.zeros((rows, cols))
    I_filtered_accum = np.zeros((rows, cols))
    I_filtered_bayer = np.zeros((rows, cols, c))

    for k in range(c):
        # select 1 channel
        IDbin = IDbin_all[:, :, k]

        # horizontal direction
        # padding borders via extra linear interpolation before filtering
        IDbin_pad_h = np.zeros((IDbin.shape[0], IDbin.shape[1] + fw - 1))
        for j in range(IDbin.shape[0]):
            # Pad left side
            [m_left, b_left] = np.polyfit(np.arange(0, (fw + 1) / 2), IDbin[j, 0:(fw + 1) // 2], 1)
            x_left = np.arange(-(fw - 1) / 2, 1)
            IDbin_pad_h[j, 0: (fw + 1) // 2] = x_left * m_left + b_left;
            # Pad right side
            [m_right, b_right] = np.polyfit(np.arange(-(fw - 1) / 2, 1), IDbin[j, -(fw - 1) // 2 - 1:], 1)
            x_right = np.arange(0, (fw + 1) / 2)
            IDbin_pad_h[j, - (fw - 1) // 2 - 1:] = x_right * m_right + b_right

        IDbin_pad_h[:, (fw + 1) // 2 - 1: - (fw - 1) // 2] = IDbin

        # 3x1 median filter (vertical)
        I_medfilt_h = spsignal.medfilt(IDbin_pad_h, [3, 1])

        # horizontal filtering
        I_filtered_h = imfilter_with_1d_kernel(I_medfilt_h, h, axis=0)
        I_filtered_h[I_filtered_h < 0] = 0

        # save a backup before median filtering
        I_filtered_h_bk = I_filtered_h

        # 3x3 median fiter on the filtered image
        I_filtered_h = spsignal.medfilt2d(I_filtered_h)

        # vertical direction
        # padding borders via extra interpolation before filtering
        IDbin_pad_v = np.zeros((IDbin.shape[0] + fw - 1, IDbin.shape[1]))
        for j in range(IDbin.shape[1]):
            # Pad top side
            [m_top, b_top] = np.polyfit(np.arange(0, (fw + 1) / 2), IDbin[0:(fw + 1) // 2, j], 1)
            x_top = np.arange(-(fw - 1) / 2, 1)
            IDbin_pad_v[0: (fw + 1) // 2, j] = x_top * m_top + b_top;
            # Pad bottom side
            [m_bottom, b_bottom] = np.polyfit(np.arange(-(fw - 1) / 2, 1), IDbin[-(fw - 1) // 2 - 1:, j], 1)
            x_bottom = np.arange(0, (fw + 1) / 2)
            IDbin_pad_v[- (fw - 1) // 2 - 1:, j] = x_bottom * m_bottom + b_bottom

        IDbin_pad_v[(fw + 1) // 2 - 1: - (fw - 1) // 2, :] = IDbin

        # 1x3 median filter (vertical)
        I_medfilt_v = spsignal.medfilt(IDbin_pad_v, [1, 3])

        # horizontal filtering
        I_filtered_v = imfilter_with_1d_kernel(I_medfilt_v, h, axis=1)
        I_filtered_v[I_filtered_v < 0] = 0

        # save a backup before median filtering
        I_filtered_v_bk = I_filtered_v

        # 3x3 median fiter on the filtered image
        I_filtered_v = spsignal.medfilt2d(I_filtered_v)

        # combine veritcal and horizontal results (edge, center, and corners are treated differently)
        I_filtered_c = (I_filtered_h + I_filtered_v) / 2  # average

        # 4 edges
        I_filtered_c[(fw + 1) // 2 - 1: - (fw - 1) // 2 - 1, 0: (fw + 1) // 2] = I_filtered_v[
                                                                                 (fw + 1) // 2 - 1: - (fw - 1) // 2 - 1,
                                                                                 0: (fw + 1) // 2]
        I_filtered_c[(fw + 1) // 2 - 1: - (fw - 1) // 2 - 1, - (fw - 1) // 2 - 1:] = I_filtered_v[(fw + 1) // 2 - 1: - (
                fw - 1) // 2 - 1, - (fw - 1) // 2 - 1:]
        I_filtered_c[0: (fw + 1) // 2, (fw + 1) // 2 - 1: - (fw - 1) // 2 - 1] = I_filtered_h[0: (fw + 1) // 2,
                                                                                 (fw + 1) // 2 - 1: - (fw - 1) // 2 - 1]
        I_filtered_c[- (fw - 1) // 2:, (fw + 1) // 2 - 1: - (fw - 1) // 2 - 1] = I_filtered_h[- (fw - 1) // 2:,
                                                                                 (fw + 1) // 2 - 1: - (fw - 1) // 2 - 1]

        # 4 corners
        I_filtered_c[0:(fw + 1) // 2, 0:(fw + 1) // 2] = np.min(
            np.dstack((I_filtered_h[0:(fw + 1) // 2, 0:(fw + 1) // 2], I_filtered_v[0:(fw + 1) // 2, 0:(fw + 1) // 2])),
            axis=2)
        I_filtered_c[0:(fw + 1) // 2, -(fw - 1) // 2 - 1:] = np.min(np.dstack(
            (I_filtered_h[0:(fw + 1) // 2, -(fw - 1) // 2 - 1:], I_filtered_v[0:(fw + 1) // 2, -(fw - 1) // 2 - 1:])),
            axis=2)
        I_filtered_c[-(fw - 1) // 2 - 1:, 0:(fw + 1) // 2] = np.min(np.dstack(
            (I_filtered_h[-(fw - 1) // 2 - 1:, 0:(fw + 1) // 2], I_filtered_v[-(fw - 1) // 2 - 1:, 0:(fw + 1) // 2])),
            axis=2)
        I_filtered_c[-(fw - 1) // 2 - 1:, -(fw - 1) // 2 - 1:] = np.min(np.dstack((I_filtered_h[-(fw - 1) // 2 - 1:,
                                                                                   -(fw - 1) // 2 - 1:],
                                                                                   I_filtered_v[-(fw - 1) // 2 - 1:,
                                                                                   -(fw - 1) // 2 - 1:])), axis=2)

        # add single color plane into bayer_image
        if k == 0:
            I_filtered_r = I_filtered_c
        elif k == 1:
            I_filtered_gr = I_filtered_c
        elif k == 2:
            I_filtered_gb = I_filtered_c
        elif k == 3:
            I_filtered_b = I_filtered_c

    I_filtered_accum = I_filtered_r + I_filtered_gr + I_filtered_gb + I_filtered_b
    I_filtered_raw[::2, ::2] = I_filtered_r
    I_filtered_raw[::2, 1::2] = I_filtered_gr
    I_filtered_raw[1::2, ::2] = I_filtered_gb
    I_filtered_raw[1::2, 1::2] = I_filtered_b
    I_filtered_bayer = np.dstack((I_filtered_r, I_filtered_gr, I_filtered_gb, I_filtered_b))

    I_filtered_bayer_after_compensated = np.zeros(I_filtered_bayer.shape)
    if compensation:
        I_filtered_bayer_after_compensated = lcb_compensation.lcb_brightness_compensation(IDbayer, I_filtered_bayer)
        I_filtered_raw[::2, ::2] = I_filtered_bayer_after_compensated[:, :, 0]
        I_filtered_raw[::2, 1::2] = I_filtered_bayer_after_compensated[:, :, 1]
        I_filtered_raw[1::2, ::2] = I_filtered_bayer_after_compensated[:, :, 2]
        I_filtered_raw[1::2, 1::2] = I_filtered_bayer_after_compensated[:, :, 3]
    # for testing
    # cv2.imshow("r", cv2.applyColorMap(rescale_intensity(I_filtered_r), cv2.COLORMAP_JET))
    # cv2.imshow("gr", cv2.applyColorMap(rescale_intensity(I_filtered_gr), cv2.COLORMAP_JET))
    # cv2.imshow("gb", cv2.applyColorMap(rescale_intensity(I_filtered_gb), cv2.COLORMAP_JET))
    # cv2.imshow("b", cv2.applyColorMap(rescale_intensity(I_filtered_b), cv2.COLORMAP_JET))
    # cv2.imshow("accum", cv2.applyColorMap(rescale_intensity(I_filtered_accum), cv2.COLORMAP_JET))
    # cv2.imshow("bayer", cv2.applyColorMap(rescale_intensity(I_filtered_raw), cv2.COLORMAP_JET))

    # rgb = bilinear_interpolation(I_filtered_raw)
    # cv2.imshow("bayer", cv2.applyColorMap(rescale_intensity(rgb), cv2.COLORMAP_JET))
    # cv2.waitKey()

    output_image = I_filtered_raw

    if interpolation:
        output_image = bilinear_interpolation(I_filtered_raw)

    if exceed2maxval:
        output_image = rescale_intensity(output_image, threshold=3.5)

    return output_image


def binning(ID_fullRes, block_size=[13, 13], block_stat="mean"):
    h = ID_fullRes.shape[0]
    w = ID_fullRes.shape[1]
    c = ID_fullRes.shape[2]

    block_size_r = block_size[0];
    block_size_c = block_size[1];

    # calculate size of binned image
    bin_size = np.array((h // block_size_r, w // block_size_c))

    # if resolution does not bin evenly, extra pixels are placed
    bin_extraPix = np.array((h, w)) - np.array(block_size) * bin_size
    bin_padR = bin_extraPix[0] // 2
    bin_padC = bin_extraPix[1] // 2

    # if the padding is opposite in number, the additional value is added towards the last edge
    # this gives the start address for subsequent rows / cols
    rows = np.hstack((0, np.arange(block_size_r + bin_padR + bin_extraPix[0] % 2, h - (block_size_r + bin_padR) + 1,
                                   block_size_r), h))
    cols = np.hstack((0, np.arange(block_size_c + bin_padC + bin_extraPix[1] % 2, w - (block_size_c + bin_padC) + 1,
                                   block_size_c), w))
    # calculate the binned image
    ID_binned = np.zeros((bin_size[0], bin_size[1], c))

    for i in range(c):
        for j in range(bin_size[1]):  # width
            for k in range(bin_size[0]):  # height
                x1 = cols[j]
                y1 = rows[k]
                x2 = cols[j + 1]
                y2 = rows[k + 1]
                cords = [x1, y1, x2, y2]  # debug for cordinates
                roiData = ID_fullRes[y1:y2, x1:x2, i]

                if block_stat is "mean":
                    ID_binned[k, j, i] = np.mean(roiData)
                elif block_stat is "median":
                    ID_binned[k, j, i] = np.median(roiData)
                elif block_stat is "mode":
                    ID_binned[k, j, i] = np.searchsorted(np.unique(roiData), roiData.flat)

    return ID_binned, bin_padR, bin_padC
