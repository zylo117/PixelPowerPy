import argparse
import datetime
import array
import numpy
from funtset import *

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the raw image")
ap.add_argument("-p", "--pedestal", type=int, default=64, help="amount of pedestal to add")
ap.add_argument("-m", "--mode", type=int, default=2, help="crop mode")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()

# unsigned integer, 16位
raw = array.array('H', open(args["image"], "rb").read())

width = raw[0]
height = raw[1]

raw = numpy.array(raw)
raw = raw[2:]

raw = raw.reshape((height, width))
print(raw[0][0], raw[0][1], raw[1][0], raw[1][1])

# 按mode切片
raw = crop_by_mode(raw, args["mode"])

# 增益
raw = raw + [args["pedestal"]]
print(raw[0][0], raw[0][1], raw[1][0], raw[1][1])

# 白平衡
raw = white_balance(raw)

# 去镜头阴影
raw = lens_shading_correction(raw, 75)

time2 = datetime.datetime.now()
print(time2 - time1)


