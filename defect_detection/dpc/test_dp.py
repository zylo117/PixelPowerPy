import datetime
import argparse
from dpc.defective_pixel import *

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
dpc, dp_pointset, ID = dp(args["imageinput"])
time2 = datetime.datetime.now()
print(time2 - time1)
print(dp_pointset)
draw_defective_pixel(dpc, ID)
