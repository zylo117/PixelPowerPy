import cv2
import numpy
from skimage import exposure
import argparse
import imutils
import random
import datetime
from jpype import *
import os
import csv


# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="path to the raw image")
#
# args = vars(ap.parse_args())
#
# time1 = datetime.datetime.now()
#
# j2p_tmp_path = os.getcwd() + "\j2p_tmp.csv"
#
# open(j2p_tmp_path, "w")
#
# jarpath = os.path.join(os.path.abspath('.'), 'Bin2Raw4Py.jar')
# startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % jarpath)
# java.lang.System.out.println(java.lang.System.getProperty("user.dir"))
# bin2raw = JPackage('readbin').Bin2Raw4Py
# b2r = bin2raw()
# raw_java = b2r.getRaw2D(args["image"], j2p_tmp_path)
# java.lang.System.out.println(raw_java[0][0])
# java.lang.System.out.println(raw_java[0][1])
#
# # raw = list(raw_java)
#
# shutdownJVM()
#
# time2 = datetime.datetime.now()
#
# print(time2 - time1)
#
# result = csv.reader(open(j2p_tmp_path, 'r'))

class b2a:
    def __init__(self, bin_path):
        self.bin_path = bin_path

    def java2py_bin2array(self):
        j2p_tmp_path = os.getcwd() + "\j2p_tmp.csv"

        open(j2p_tmp_path, "w")

        # read from java
        jarpath = os.path.join(os.path.abspath('.'), 'Bin2Raw4Py.jar')
        startJVM(getDefaultJVMPath(), "-ea", "-Djava.class.path=%s" % jarpath)
        bin2raw = JPackage('readbin').Bin2Raw4Py
        b2r = bin2raw()
        raw_java = b2r.getRaw2D(self.bin_path, j2p_tmp_path)
        shutdownJVM()

        result = csv.reader(open(j2p_tmp_path, 'r'))

        result = numpy.array(tuple(result))
        # result = numpy.loadtxt(open(j2p_tmp_path, "r"), dtype=numpy.str, delimiter=",", skiprows=0)

        return result
