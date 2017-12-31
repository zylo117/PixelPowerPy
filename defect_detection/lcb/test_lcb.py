import datetime
import argparse
import timeit
import imutils
import numpy as np
import cv2
from lcb.low_contrast_blemish import lcb
from io_bin.preprocess import preprocess

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--imageinput", required=True, help="path to the raw image")
ap.add_argument("--inputbayerformat", type=str, default="rggb", help="bayerformat of input image data")
ap.add_argument("-o", "--outputformat", type=str, default="raw", help="format of output image data")
ap.add_argument("-p", "--pedestal", type=int, default=64, help="amount of pedestal to add")
ap.add_argument("-m", "--mode", type=int, default=2, help="crop mode")
ap.add_argument("-f", "--FOV", type=int, default=0, help="Field of view")
ap.add_argument("-w", "--whitebalance", type=bool, default=True, help="whether apply whitebalance")
ap.add_argument("-b", "--bitdepth", type=int, default=10, help="depth of the color")
ap.add_argument("-s", "--signed", type=bool, default=True, help="Whether all pixels value will be signed")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()
# custom_source = preprocess(imageinput=args["imageinput"], outputformat="bayer", more_precise=True, custom_size=[3856, 2340], custom_decoding="B")
ID = lcb(args["imageinput"], compensation=False, mode=0, roiSize=[9, 9], threshold=3.1)
time2 = datetime.datetime.now()
print(time2 - time1)

resized = imutils.resize(ID, width=1024)
heat_map = cv2.applyColorMap(resized, cv2.COLORMAP_JET)
max = np.where(resized == np.max(resized))

for i in range(len(max)):
    cv2.circle(heat_map, (max[1][i], max[0][i]), 20, (128, 128, 128), 3)
cv2.imshow("LCB", heat_map)
cv2.waitKey()
# t1 = timeit.Timer(lambda: lcb(args["imageinput"]))
# print(timeit.timeit())
