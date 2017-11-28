import cv2
import numpy
from skimage import exposure
import argparse
import imutils
import auto_canny
import random

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the raw image")
ap.add_argument("-b", "--bayer", default=4, help="Bayer format")
ap.add_argument("-p" "--pedestal", default=64, help="the base value that will add to image at first")

args = vars(ap.parse_args())

bayer_format = args["bayer"]
pedestal = args["pedestal"]
