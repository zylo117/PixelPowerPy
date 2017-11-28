import argparse
import datetime
import pyximport;pyximport.install()
import bin2array_c

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the raw image")

args = vars(ap.parse_args())

raw_dec = [];

time1 = datetime.datetime.now()

raw_dec = bin2array_c.b2a(args["image"])

time2 = datetime.datetime.now()

print(time2 - time1)


