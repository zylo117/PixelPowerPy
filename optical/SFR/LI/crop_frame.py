# cropFrame_LI.m crop raw images from integrator, correct PDAF gain/crosstalk pixels
# and adjust image resolution to the testing resolution needed
# INPUT:
#   IDraw (4 options of resolution)
#       1. 3552 x 2440
#       2. 3376 x 2324
#       3. 3408 x 2324
#       4. 3120 x 2324
#   bayerFormat = 'rggb'
#   bitDepth = 10
#   program = 'long island'
#   pixelType = 'normal','mono' -> Do nothing
#             = 'oclb' -> OCL-B type PDAF pattern
#   isFPCXTlk = 1 -> correcting Xtlk pixels
#             = 0 -> no correction
#   isFPCGain = 1 -> correcting Gain pixels
#             = 0 -> no correction
#   isCrop = 1 -> cropping image to desired resoluiton
#          = 0 -> no cropping shall be applied
#
# OUTPUT:
#   ID: image array as requested
import numpy as np

from LI.focusGains_OCL_xtlk_Comp import focusGains_OCL_xtlk_Comp


def cropFrame_LI(IDraw, bayerFormat, bitDepth, program, pixelType, isFPCXTlk, isFPCGain, isCrop):
    # Update PDAF correction offset based on MI captured image
    # Updated to 4.5 - option to disable xtalk compensation for SFR tests for
    # ES1 sensor
    # Updated to 4.6 - Full resolution image capture support for LI 3552x2440
    roiSize = []
    ROIs = []
    Kernels = []
    offset = []
    imgHeight, imgWidth = IDraw.shape[:2]
    imgRes_rev = 0

    if program=='li':
        pedestal = -16
        IDrawComp = IDraw
        if (imgHeight == 2440) and (imgWidth == 3552):  # full resolution output from sensor
            imgRes_rev = 1
        elif (imgHeight == 2324) and (imgWidth == 3376):  # 3376 x 2324 resolution is for LightField & Dark tests
            imgRes_rev = 2
            isCrop = 0
        elif (imgHeight == 2324) and (imgWidth == 3408):  # 3408 x 2324 resolution is for LightField & Dark tests
            imgRes_rev = 3
        elif (imgHeight == 2324) and (imgWidth == 3120):  # 3120 x 2324 resolution is for SFR tests matching 68deg FOV
            imgRes_rev = 4
        else:
            print('unknown resolution for gain/xtlk compensation')
            return

    if pixelType == "normal":
        pass
    elif pixelType == "mono":
        pass
    elif pixelType == "oclb":
        if imgRes_rev == 1:
            ROIsX = 220
            ROIsY = 150  # Yonkers 3552 x 2440, 16pix border 16x16 cell
            offsetX = 17
            offsetY = 17
            medianAreaX_Gain = 220
            medianAreaY_Gain = 400
            medianAreaX_xtlk = 440
            medianAreaY_xtlk = 400
            # OCL - B
            Kernels = np.zeros([4, 2])
            Kernels[0, :] = [3, 2]
            Kernels[1, :] = [11, 2]
            Kernels[2, :] = [3, 10]
            Kernels[3, :] = [11, 10]

        elif imgRes_rev == 2:
            ROIsX = 211
            ROIsY = 145  # Yonkers 3376 x 2324 cropped 16x16 cell
            offsetX = 1
            offsetY = 1
            medianAreaX_Gain = 211
            medianAreaY_Gain = 387
            medianAreaX_xtlk = 422
            medianAreaY_xtlk = 387
            # OCL - B
            Kernels = np.zeros([4, 2])
            Kernels[0, :] = [1, 2]
            Kernels[1, :] = [9, 2]
            Kernels[2, :] = [1, 10]
            Kernels[3, :] = [9, 10]

        elif imgRes_rev == 3:
            ROIsX = 213
            ROIsY = 145  # Yonkers 3408 x 2324 cropped 16x16 cell
            offsetX = 1
            offsetY = 1 + 6
            medianAreaX_Gain = 213
            medianAreaY_Gain = 387
            medianAreaX_xtlk = 426
            medianAreaY_xtlk = 387
            # OCL - B
            Kernels = np.zeros([4, 2])
            Kernels[0, :] = [1, 2]
            Kernels[1, :] = [9, 2]
            Kernels[2, :] = [1, 10]
            Kernels[3, :] = [9, 10]

        elif imgRes_rev == 4:
            ROIsX = 195
            ROIsY = 145  # Yonkers 3120 x 2324 cropped 16x16 cell
            offsetX = 1
            offsetY = 1 + 6
            medianAreaX_Gain = 195
            medianAreaY_Gain = 387
            medianAreaX_xtlk = 390
            medianAreaY_xtlk = 387
            # OCL - B
            Kernels = np.zeros([4, 2])
            Kernels[0, :] = [1, 2]
            Kernels[1, :] = [9, 2]
            Kernels[2, :] = [1, 10]
            Kernels[3, :] = [9, 10]

        roiX = 16
        roiY = 16
        roiSize = [roiX, roiY]
        ROIs = [ROIsX, ROIsY]
        offset = [offsetX, offsetY]
    else:
        print("Invalid pixel type detected")

    # Correct Gain and/or cross talk pixels
    if pixelType == "oclb" and (isFPCGain or isFPCXTlk):
        IDrawComp = focusGains_OCL_xtlk_Comp(IDraw, bayerFormat, pedestal,
                                             bitDepth, roiSize, ROIs, Kernels, offset, isFPCGain, isFPCXTlk)
    else:
        IDrawComp = IDraw
    # Image cropping processing should be AFTER gain/xtlk compensation
    if isCrop:
        if imgRes_rev == 3:  # 3376 x 2324 resolution is for LightField & Dark tests
            cropPixel_Height = 0
            cropPixel_Width = 16
            IDrawComp = IDrawComp[cropPixel_Height:imgHeight - cropPixel_Height, cropPixel_Width:imgWidth - cropPixel_Width]
        elif imgRes_rev == 4:  # 3098 x 2324 resolution is for SFR tests matching 68deg FOV
            cropPixel_Height = 0
            cropPixel_Width = 11
            IDrawComp = IDrawComp[cropPixel_Height:imgHeight - cropPixel_Height, cropPixel_Width - 1:imgWidth - cropPixel_Width - 1]  # note that shift 1 column to left
        elif imgRes_rev == 1:  # 3376 x 2324 resolution for D50, A and Dark tests that require cropped resolution
            cropPixel_Height = 58
            cropPixel_Width = 88
            IDrawComp = IDrawComp[cropPixel_Height:imgHeight - cropPixel_Height, cropPixel_Width:imgWidth - cropPixel_Width]
    else:
        # do not crop
        print("full sensor resolution output")

    return IDrawComp