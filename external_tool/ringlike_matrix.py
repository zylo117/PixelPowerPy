import cv2

import numpy as np


class RingMat:
    def __init__(self, centrosymmetry_matrix, radius=None, dtype=np.uint8):
        """

        :param centrosymmetry_matrix:
        :param radius: radius = shorter length / 2, inscribed	circle; radius = diagonal / 2, circumscribed circle
        :param dtype:
        """

        self.sqmat = centrosymmetry_matrix
        self.radius = centrosymmetry_matrix.shape[0] // 2 if radius is None else radius
        self.data = []  # init data 2D List (List -> numpy array)
        self.dtype = dtype

        self.get_data()

    def get_data(self):
        center = (self.sqmat.shape[1] // 2, self.sqmat.shape[0] // 2)
        for i in range(self.radius):  # add one ring as padding
            mask = np.zeros(self.sqmat.shape).astype(self.dtype)
            cv2.circle(mask, center, i, (255, 255, 255), 1)
            ring = cv2.bitwise_and(self.sqmat, mask)
            ring = ring[ring > 0]
            self.data.append(ring)

    def mean(self):
        mean = np.zeros(self.radius)
        for i in range(len(self.data)):
            mean[i] = np.mean(self.data[i])
        return mean

    def std(self):
        std = np.zeros(self.radius)
        for i in range(len(self.data)):
            std[i] = np.std(self.data[i])
        return std

    def min(self):
        min = np.zeros(self.radius)
        for i in range(len(self.data)):
            min[i] = np.min(self.data[i])
        return min

    def max(self):
        max = np.zeros(self.radius)
        for i in range(len(self.data)):
            max[i] = np.max(self.data[i])
        return max

    # super hard core
    def recreate_ringmat(self, specified_data=None, size=None):
        img = np.zeros(self.sqmat.shape).astype(self.dtype) if size is None else np.zeros(size).astype(self.dtype)
        data = self.mean() if specified_data is None else specified_data

        center = (img.shape[1] // 2, img.shape[0] // 2)
        for i in range(self.radius):
            cv2.circle(img, center, i, (data[i], data[i], data[i]), 1)

        # result may include plenty of black dots, median blur it to cover them
        img = cv2.medianBlur(img, 3)
        # black_dots = np.where(img == 0)
        # w = 3
        # for i in range(len(black_dots[0])):
        #     print(black_dots[0][i],black_dots[1][i])
        #     img[black_dots[0][i],black_dots[1][i]] = np.nanmean(img[black_dots[0][i] - 1: black_dots[0][i] + 2, black_dots[1][i] - 1: black_dots[1][i] + 2]).astype(np.uint8)

        return img
