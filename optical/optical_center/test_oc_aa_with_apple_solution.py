import datetime
import argparse
from optical_center.optical_center_active_alignment import *
import imutils

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--imageinput", required=True, help="path to the raw image")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()
custom_source = preprocess(imageinput=args["imageinput"], outputformat="yuv", more_precise=True, custom_size=[3856, 2340], custom_decoding="B", FOV=75)
oc, ID = oc(args["imageinput"], custom_source=custom_source)
# draw_optical_center(oc, ID)
print(oc)

ID_rgb = cv2.cvtColor((ID / 4).astype(np.uint8), cv2.COLOR_YUV2BGR)
_, thresh = cv2.threshold(ID_rgb, oc[0] / 4, 255, cv2.THRESH_BINARY)

cv2.imshow("Thresh", imutils.resize(thresh, width=800))
cv2.waitKey()

time2 = datetime.datetime.now()
print(time2 - time1)
