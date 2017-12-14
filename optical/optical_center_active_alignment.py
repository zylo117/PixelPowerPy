import numpy as np
import cv2
import imutils
from auto_canny import auto_canny
from skimage import exposure
from io_bin.preprocess import preprocess


def oc_aa(IDraw, detect_thresh_val = 5, center_area_percentage = 0.3, bayerformat="rggb", pedestal=64, bitdepth=10, custom_size=[3856, 2340]):
    raw_rgb = preprocess(IDraw, outputformat="rgb", more_precise=True, custom_size=[3856, 2340], custom_decoding="B")
    raw_rgb = (raw_rgb / 4).astype(np.uint8)
    raw_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY)

    height, width = raw_gray.shape[:2]

    print(raw_gray[1169, 1927])
    print(np.min(raw_gray))
    print(np.max(raw_gray))
    black_dot_map = (raw_gray <= (np.min(raw_gray) + detect_thresh_val)).astype(np.uint8)
    black_dot_map[black_dot_map != 0] = 255

    black_dot_edge = auto_canny(black_dot_map)

    cnts= cv2.findContours(black_dot_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
    cv2.drawContours(raw_rgb, cnts, -1, (255, 0, 255), -1)

    cv2.imshow("Thresh", imutils.resize(raw_rgb, width=800))

    return raw_gray
