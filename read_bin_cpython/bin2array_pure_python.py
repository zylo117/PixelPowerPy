import argparse
import datetime
import array
import numpy

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the raw image")

args = vars(ap.parse_args())
# raw = open(args["image"], "rb").read()
raw_dec = [];

time1 = datetime.datetime.now()

raw = array.array('H', open(args["image"], "rb").read())

time2 = datetime.datetime.now()

print(time2 - time1)

width = raw[0]
height = raw[1]
print(width, height, raw[0], raw[1], raw[2], raw[3])

raw = numpy.array(raw)
raw = raw[2:]
print(raw[0], raw[1], raw[2], raw[3])

raw = raw.reshape((height,width))
print(raw[0][0], raw[0][1], raw[1][0], raw[1][1])
