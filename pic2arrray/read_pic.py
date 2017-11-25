import cv2
import numpy
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-s", "--kernel_size", default=4, help="size of the kernel, the bigger, the brighter, but the blurrier")
ap.add_argument("-o" "--output_image_path", help="path to the output image")
args = vars(ap.parse_args())

