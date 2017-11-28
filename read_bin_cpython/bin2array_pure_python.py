import argparse
import datetime
import numpy
import pyximport;

pyximport.install()
import bin2array_c

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the raw image")

args = vars(ap.parse_args())

raw_dec = [];

time1 = datetime.datetime.now()

raw_dec = bin2array_c.b2a(args["image"])
print(len(raw_dec))

time2 = datetime.datetime.now()

print(time2 - time1)

width = raw_dec[0]
height = raw_dec[1]

raw_dec = numpy.array(raw_dec)

raw_dec = raw_dec[2:]

raw_dec = raw_dec.reshape((height, width))

print(width, height)
print(raw_dec[0][0], raw_dec[0][1], raw_dec[1][0], raw_dec[1][1])
