import datetime

import cv2
import imutils
import numpy as np
import scipy as sp
import scipy.signal as spsignal

from read_bin import BinFile


class LCB:
    def __init__(self, raw_img):
        self.raw_img = raw_img
        self.heatmap = None
        self.dust_stat = []

        self.shape = self.raw_img.shape
        if len(self.shape) == 2:
            self.src_type = "raw"
        elif len(self.shape) == 3:
            if self.shape[2] == 3:
                self.src_type = "rgb"
            elif len(self.shape) == 4:
                self.src_type = "bayer"

    def binning(self, roi_size, kernel_size):
        # must use cv2.INTER_AREA as interpolation
        if self.src_type == "raw":
            plane = np.zeros((self.raw_img.shape[0] // 2, self.raw_img.shape[1] // 2, 4)).astype(np.float32)
            plane[:, :, 0] = self.raw_img[::2, ::2]
            plane[:, :, 1] = self.raw_img[::2, 1::2]
            plane[:, :, 2] = self.raw_img[1::2, ::2]
            plane[:, :, 3] = self.raw_img[1::2, 1::2]

            return cv2.resize(plane, (plane.shape[1] // roi_size, plane.shape[0] // roi_size),
                              interpolation=cv2.INTER_AREA)
        else:
            return cv2.resize(self.raw_img, (self.shape[1] // roi_size, self.shape[0] // roi_size),
                              interpolation=cv2.INTER_AREA)

    def LCB_compact(self, roi_size=13, kernel_size=9, usl=11.1, debug=False):
        bin_img = self.binning(roi_size, kernel_size)
        [rows, cols, c] = bin_img.shape
        img_heatmap_accumulation = np.zeros((rows, cols)).astype(np.float32)

        # define kernel
        fw = kernel_size
        kernel = np.hstack(
            (1 / 2, np.zeros((fw - 3) // 2), -1, np.zeros((fw - 3) // 2), 1 / 2))

        for k in range(c):
            # select 1 channel
            img = bin_img[:, :, k]

            img_filtered_h = cv2.filter2D(img, cv2.CV_32F, kernel, borderType=cv2.BORDER_ISOLATED)
            img_filtered_h = cv2.medianBlur(img_filtered_h, 3)
            img_filtered_v = np.transpose(
                cv2.filter2D(np.transpose(img), cv2.CV_32F, kernel, borderType=cv2.BORDER_ISOLATED))
            img_filtered_v = cv2.medianBlur(img_filtered_v, 3)

            # img_filtered_h[img_filtered_h < 0] = 0
            # img_filtered_v[img_filtered_v < 0] = 0

            img_filtered = (img_filtered_h + img_filtered_v) / 2

            img_filtered = cv2.medianBlur(img_filtered.astype(np.float32), 3)

            img_filtered[img_filtered < 0] = 0

            img_heatmap_accumulation = img_heatmap_accumulation + img_filtered

        img_heatmap_accumulation = img_heatmap_accumulation * 255 / usl
        img_heatmap_accumulation[img_heatmap_accumulation > 255] = 255
        img_heatmap_accumulation = img_heatmap_accumulation.astype(np.uint8)
        self.heatmap = cv2.applyColorMap(img_heatmap_accumulation, cv2.COLORMAP_JET)

        if debug:
            cv2.imshow("Test", cv2.resize(self.heatmap, (600, 400)))
            cv2.waitKey(0)

    def dust_analysis(self, custom_lcb_img=None, debug=False):
        heatmap = self.heatmap if custom_lcb_img is None else custom_lcb_img

        R = heatmap[:, :, 2].astype(np.uint16)  # R means LCB points
        G = heatmap[:, :, 1].astype(np.uint16)  # G means it is near LCB points

        defects = cv2.convertScaleAbs(R + G)

        for i in range(9):
            defects = cv2.medianBlur(defects, 3)

        defects = imutils.skeletonize(defects, (3, 3), cv2.MORPH_ELLIPSE)
        defects[defects > 0] = 255

        cnts = cv2.findContours(defects.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        for c in cnts:
            center, radius = cv2.minEnclosingCircle(c)
            point_in_contour = cv2.pointPolygonTest(c, center, False)

            dust_type = None
            if point_in_contour < 0:
                dust_type = "Lens"
                self.dust_stat.append([center, radius, dust_type])
            elif R[int(center[1]), int(center[0])] > 0:
                dust_type = "IRCF"
                self.dust_stat.append([center, radius, dust_type])
            # else:
            #     dust_type = "Flare"
            #     self.dust_stat.append([center, radius, dust_type])

            if debug:
                cv2.circle(heatmap, (int(center[0]), int(center[1])), int(radius), (255, 0, 255), 1)
                cv2.putText(heatmap, dust_type, (int(center[0]) + int(radius), int(center[1] - int(radius))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255))

        if debug:
            print(np.count_nonzero(R))

            cv2.imshow("defects", cv2.resize(defects, (600, 400)))

            # cv2.drawContours(heatmap, cnts, -1, (255, 0, 255), 1)
            cv2.imshow("heatmap", cv2.resize(heatmap, (600, 400)))

            cv2.waitKey(0)


if __name__ == "__main__":
    time1 = datetime.datetime.now()
    bin = BinFile("/Volumes/OSX_Data/tmp/0507/LensShadingQuad_BB_20180428054817_FX8816500ASJGCP3G_C575145115D7E803_APC.raw")

    a = bin.get_realdata(2)[:20]
    (width, height), bindata = a

    raw_img = bindata.reshape((height, width))
    # for i in range(30):
    #     lcb_img = LCB(raw_img)
    #     lcb_img.LCB_compact(usl=11.1, debug=0)
    #     lcb_img.dust_analysis(debug=False)

    lcb_img = LCB(raw_img)
    lcb_img.LCB_compact(roi_size=13, kernel_size=9, usl=11.1, debug=False)
    lcb_img.dust_analysis(debug=True)

    time2 = datetime.datetime.now()
    print(time2 - time1)
