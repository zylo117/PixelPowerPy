import cv2
import numpy
from skimage import exposure
import argparse
import imutils
import random
import datetime
from jpype import *
from bin2array_from_java import b2a

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the raw image")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()

bin = b2a(args["image"])

raw = bin.java2py_bin2array()

time2 = datetime.datetime.now()

print(time2 - time1)

print(raw[0][0])

width = raw.shape[1] - 2
height = raw.shape[0]

print(raw[0][0], width, height)

raw = raw + 64
print(raw[0][0])
