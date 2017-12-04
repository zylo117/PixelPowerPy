import datetime
import argparse
from optical.optical_center import *

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--imageinput", required=True, help="path to the raw image")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()
oc, ID = oc(args["imageinput"])
time2 = datetime.datetime.now()
print(time2 - time1)
print("x: ", oc[3], "   y: ", oc[4], "  magShift: ", oc[5])
draw_optical_center(oc, ID)
