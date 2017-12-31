import datetime
import argparse
from optical.illumination.relative_illumination import *

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--imageinput", required=True, help="path to the raw image")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()
ri = ri(args["imageinput"])
time2 = datetime.datetime.now()
print(time2 - time1)

print(ri)
