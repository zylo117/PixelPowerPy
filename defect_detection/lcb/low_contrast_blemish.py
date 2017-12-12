import numpy as np
import numba
from preprocess import preprocess


def lcb(IDraw, bayerformat="rggb", pedestal=64, bitdepth=10, roiSize=[13, 13], filterWidth=9, threshold=12.6):
    IDbayer = preprocess(IDraw, outputformat="bayer", mode=2)

    height = IDbayer.shape[0] * 2
    width = IDbayer.shape[1] * 2

    roiX_size = roiSize[1]
    roiY_size = roiSize[0]
    fw = filterWidth

    CornerROISize = [filterWidth + 1, filterWidth + 1]

    cornerX_size = CornerROISize[1]
    cornerY_size = CornerROISize[0]

    # define the filter
    h = np.hstack((1 / 2, np.zeros((fw - 3) // 2), -1, np.zeros((fw - 3) // 2), 1 / 2))

    # scale down input image

    IDbin_all = binning(IDbayer);
    [rows, cols, c] = IDbin_all.shape

    # for

    print(0)


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

    return ID_binned
