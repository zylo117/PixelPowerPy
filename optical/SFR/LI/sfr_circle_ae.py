# Calculating AE values based on SFR circle chart
# AE: Auto Exposure
# the output = AEVal shoud meet target of 75% +/-5%  * 1023: LSL= 716, USL=818

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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
    pix = np.sort(roi_tmp)
    numPix = len(pix)
    idx85 = int(np.round(0.85 * numPix))
    AEVal = pix[idx85]

    AE = {}

    AE["value"]= AEVal
    AE["bitDepth"] = bitDepth
    AE["percentage"] = np.round(100 * AE["value"] / (np.power(2, bitDepth) - 1))
    AE["positionY"] = h / 2 - roiSize / 2 - 1
    AE["positionX"] = w / 2 - roiLocation - roiSize / 2 - 1
    AE["data"] = roi

    # plot roi histogram
    plt.figure()
    plt.subplot(121)
    plt.imshow(ID_Gmean, cmap="gray")
    plt.gca().add_patch(patches.Rectangle((AE["positionX"], AE["positionY"]), roiSize, roiSize, linewidth=1, edgecolor="b", fill=False))
    plt.text(w / 2 - roiLocation - roiSize / 4 - 1, h / 2 - roiSize / 4 - 1, str(AE["value"]), color="b", fontsize=6)
    plt.text(w / 2 - roiLocation - roiSize / 4 - 1, h / 2 + roiSize / 4 - 1, "Roi size: " + str(np.round(roiSize, 1)) + " X " + str(np.round(roiSize, 1)), color="r", fontsize=6)
    plt.text(w / 2 - roiLocation - roiSize / 4 - 1, h / 2 + 1.5 * roiSize / 4 - 1, "Roi location: (" + str(np.round(w / 2 - 1, 1)) + "," + str(np.round(h / 2 - roiLocation - 1, 1)) + ")", color="r", fontsize=6)
    plt.title("Roi specification and calculated AE values")

    plt.subplot(122)
    plt.plot(np.arange(len(pix)), pix, zorder=1, label="histogram")
    plt.grid(True, which="both", axis="both", color='grey', linewidth='0.3', linestyle='--')
    plt.vlines(idx85, 0, np.max(pix), colors="r", zorder=2, label="85% idx")
    plt.hlines(AE["value"], 0, len(pix), colors="g", zorder=3, label="AEVal")
    plt.title("AE @ 85% histogram of ROI")
    plt.xlabel("Pixel Index")
    plt.ylabel("Pixel Value (LSB)")  # Least Significant Bits
    plt.legend()
    plt.show()