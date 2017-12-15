import numpy as np
import cv2
import imutils
from auto_canny import auto_canny
from external_tool import math_tool
from io_bin.preprocess import preprocess


class ActiveAlignment:

    def __init__(self, raw_file):
        self.raw_file = raw_file
        self.black_dot = np.zeros((5, 2), dtype=np.double)
        self.height = 0
        self.width = 0

        # effective focal length, cm
        self.efl = 37

    def black_dot_location(self, detect_thresh_val=5, center_area_percentage=0.3, bayerformat="rggb", pedestal=64,
                           bitdepth=10, custom_size=[3856, 2340], debug=False):
        raw_rgb = preprocess(self.raw_file, outputformat="rgb", more_precise=True, custom_size=[3856, 2340],
                             custom_decoding="B")
        raw_rgb = (raw_rgb / 4).astype(np.uint8)
        raw_gray = cv2.cvtColor(raw_rgb, cv2.COLOR_BGR2GRAY)

        self.height, self.width = raw_gray.shape[:2]

        black_dot_map = (raw_gray <= (np.min(raw_gray) + detect_thresh_val)).astype(np.uint8)
        black_dot_map[black_dot_map != 0] = 255

        black_dot_edge = auto_canny(black_dot_map)

        cnts = cv2.findContours(black_dot_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        # cv2.drawContours(raw_rgb, cnts, -1, (255, 0, 255), -1)

        # center, upperleft, upperright, lowerright, lowerleft
        black_dot_ori = []

        for c in cnts:
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if debug:
                cv2.circle(raw_rgb, (int(x), int(y)), int(radius), (255, 0, 255), 2)

            black_dot_ori.append((x, y))
            # print(radius)

        black_dot_ori = np.array(black_dot_ori)
        # determine the corners
        # center
        delta_width = np.abs(black_dot_ori[:, 0] - self.width / 2)
        self.black_dot[0] = black_dot_ori[np.argmin(delta_width)]

        # ul and lr
        s = black_dot_ori.sum(axis=1)
        self.black_dot[1] = black_dot_ori[np.argmin(s)]
        self.black_dot[3] = black_dot_ori[np.argmax(s)]

        # ll and ur
        diff = np.diff(black_dot_ori, axis=1)
        self.black_dot[2] = black_dot_ori[np.argmin(diff)]
        self.black_dot[4] = black_dot_ori[np.argmax(diff)]

        if debug:
            for i in range(len(self.black_dot)):
                cv2.putText(raw_rgb, str(i + 1), (int(self.black_dot[i][0]), int(self.black_dot[i][1])), cv2.FONT_HERSHEY_SIMPLEX,
                            2,
                            (255, 0, 255), 5)

            cv2.imshow("Diagram", imutils.resize(raw_rgb, width=800))

        return self.black_dot, raw_rgb

    def oc(self):
        return [self.black_dot[0][0] - (self.width / 2 - 1), self.black_dot[0][1] - (self.height / 2 - 1)]

    def rotation_angle(self):
        # counterclockwise direction is position angle

        upper_slope = (self.black_dot[2][1] - self.black_dot[1][1]) / (self.black_dot[2][0] - self.black_dot[1][0])
        lower_slope = (self.black_dot[3][1] - self.black_dot[4][1]) / (self.black_dot[3][0] - self.black_dot[4][0])

        return np.arctan((upper_slope+lower_slope) / 2)


    def tilt_angle(self):
        """
        example:
        -->horizonal_tilt

            horizonal_length
            <------------>
            ○ left_line  ○ right_line
         ↑   --------------
         |   |            |
         |   |            |h2
         |   |            --
      efl|   | h1=efl    *|
         |   |        *   |
         |   |     *      |h3
         |   |  *         |
         ↓   *_)θ_________


        -> h2 = h1 / (left_line_length / right_line_length)
        -> h3 = h1 - h2
        ->tilt_h = h3 / horizonal_length
        """

        # counterclockwise direction is position angle

        up_line_length = math_tool.point_distance(self.black_dot[2], self.black_dot[1])
        down_line_length = math_tool.point_distance(self.black_dot[3], self.black_dot[4])
        left_line_length = math_tool.point_distance(self.black_dot[4], self.black_dot[1])
        right_line_length = math_tool.point_distance(self.black_dot[3], self.black_dot[2])

        horizonal_length = (up_line_length + down_line_length) / 2
        vertical_length = (left_line_length + right_line_length) / 2

        tilt_h = (self.efl - (self.efl / (left_line_length / right_line_length))) / horizonal_length
        tilt_v = (self.efl - (self.efl / (up_line_length / down_line_length))) / vertical_length

        return [tilt_h, tilt_v]
