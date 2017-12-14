import datetime
import argparse
import cv2
from optical.optical_center import *
from optical.optical_center_active_alignment import *

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--imageinput", required=True, help="path to the raw image")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()
# custom_source = preprocess(imageinput=args["imageinput"], outputformat="yuv", more_precise=True, custom_size=[3856, 2340], custom_decoding="B")
# oc, ID = oc(args["imageinput"], custom_source=custom_source)
ID = oc_aa(args["imageinput"])
# ID = (ID / 4).astype(np.uint8)
# ID = cv2.cvtColor(ID, cv2.COLOR_YUV2BGR)
cv2.imshow("Raw_RGB", imutils.resize(ID, width=800))
cv2.waitKey()
time2 = datetime.datetime.now()
print(time2 - time1)
# print("x: ", oc[3], "   y: ", oc[4], "  magShift: ", oc[5])
# draw_optical_center(oc, ID, magnification=30)
