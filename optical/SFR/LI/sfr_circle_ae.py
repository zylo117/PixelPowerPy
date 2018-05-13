# Calculating AE values based on SFR circle chart
# AE: Auto Exposure
# the output = AEVal shoud meet target of 75% +/-5%  * 1023: LSL= 716, USL=818

import numpy as np


def SFRCircle_AE(IDraw, bitDepth):
    # get GR & GB channel mean for AE purpose
    ID_Gmean = np.round((IDraw[0::2, 1::2] + IDraw[1::2, 0::2]) / 2).astype(np.uint)
    h, w = ID_Gmean.shape[:2]

    # adjust roiSize, roiLocation based on image diagnol 
    halfdiagnol = 0.5 * np.sqrt(np.power(h, 2) + np.power(w, 2))
    roiSize = 0.06 * halfdiagnol
    roiLocation = 0.10 * halfdiagnol  # 0.1* half diagnol shift to left side of image center

    # define roi on the image
    roi = ID_Gmean[int(np.round(h / 2 - roiSize / 2)) - 1:int(np.round(h / 2 + roiSize / 2)),
          int(np.round(w / 2 - roiLocation - roiSize / 2)) - 1:int(np.round(w / 2 - roiLocation + roiSize / 2))]

    # get 85# AEval from roi histogram
    roi_tmp = np.ravel(roi, order="F")
    idx = np.argsort(roi_tmp)
    pix = np.sort(roi_tmp)
    numPix = len(pix)
    idx85 = int(np.round(0.85 * (numPix)))
    AEVal = pix[idx85]

    AE = {}

    AE["value"]= AEVal
    AE["bitDepth"] = bitDepth
    AE["percentage"] = np.round(100 * AE["value"] / (np.power(2, bitDepth) - 1))
    AE["positionY"] = h / 2 - roiSize / 2 - 1
    AE["positionX"] = w / 2 - roiLocation - roiSize / 2 - 1
    AE["data"] = roi

    print()