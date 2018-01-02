import datetime
import argparse
from optical.illumination.diagonal_illumination_distribution import *

ap = argparse.ArgumentParser()

ap.add_argument("-i", "--imageinput", required=True, help="path to the raw image")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()
di, back_di = di(args["imageinput"])
# fi = fit_curve(di)
time2 = datetime.datetime.now()
print(time2 - time1)

draw_diag_illumination(di, back_di)


