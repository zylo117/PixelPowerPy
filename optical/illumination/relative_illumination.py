"""
relative illumination calculates the colour cal data
make sure if the illumination of four corners is similar
detect Lens shading defects

INPUT:
  IDraw: raw image data
  bayerFormat: bayer format:
      'bggr'
      'rggb'
      'grbg'
      'gbrg'

OUTPUT:
  1, relative illumination of TL, TR, BL, BR, delta of the max and min ri
"""

from io_bin.preprocess import preprocess
import numpy as np


def ri(ID, bayerformat="rggb", pedestal=64, bitdepth=10, custom_source=None):
    if custom_source is None:
        IDyuv = preprocess(ID, bayerformat=bayerformat, pedestal=pedestal, bitdepth=bitdepth, outputformat="yuv")
    else:
        IDyuv = custom_source

    IDy = IDyuv[:, :, 0]

    h, w = IDy.shape

    roiSize = 0.025

    # define centre locations for ROI
    roiCentreX = [0.5, 0.0125, 0.9875, 0.0125, 0.9875]
    roiCentreY = [0.5, 0.0125, 0.0125, 0.9875, 0.9875]

    # generate data for all ROIs
    ri = np.zeros(len(roiCentreX))
    ri_mean = []

    for i in range(len(ri)):
        x1 = int(np.round(w * roiCentreX[i] - (w * roiSize / 2) + 0.5))
        x2 = int(np.round(w * roiCentreX[i] + (w * roiSize / 2) - 1.4999))
        y1 = int(np.round(h * roiCentreY[i] - (h * roiSize / 2) + 0.5))
        y2 = int(np.round(h * roiCentreY[i] + (h * roiSize / 2) - 1.4999))
        ri_mean.append(np.mean(np.mean(IDy[y1:y2 + 1, x1: x2 + 1])))

    ri = 100 * (ri_mean / ri_mean[0])
    ri_delta = np.max(ri[1:5]) - np.min(ri[1:5])

    return ri, ri_delta
