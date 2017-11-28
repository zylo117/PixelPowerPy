import argparse
import datetime
import bin2array_c

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the raw image")

args = vars(ap.parse_args())

raw_dec = [];

time1 = datetime.datetime.now()

b2a(args["image"])

time2 = datetime.datetime.now()

print(len(raw_dec), raw_dec[1], raw_dec[0])
print(time2 - time1)
