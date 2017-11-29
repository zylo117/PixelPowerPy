import argparse
import datetime
import array
import numpy
import preprocess

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the raw image")
ap.add_argument("-p", "--pedestal", type=int, default=64, help="amount of pedestal to add")
ap.add_argument("-m", "--mode", type=int, default=2, help="crop mode")

args = vars(ap.parse_args())

time1 = datetime.datetime.now()

# unsigned integer, 16位
raw = array.array('H', open(args["image"], "rb").read())

width = raw[0]
height = raw[1]

raw = numpy.array(raw)
raw = raw[2:]

raw = raw.reshape((height, width))
print(raw[0][0], raw[0][1], raw[1][0], raw[1][1])

# 按mode切片
if args["mode"] is 1:
    raw = raw[8:height - 8, 8:width - 8]
elif args["mode"] is 2:
    raw = raw[int((height - (width - 16) / (16 / 9)) / 2): height - int((height - (width - 16) / (16 / 9)) / 2),
          8:width - 8]
elif args["mode"] is 3:
    raw = raw[8:height - 8,
          int((width - (height - 16) * (3 / 2)) / 2): width - int((width - (height - 16) * (3 / 2)) / 2)]
elif args["mode"] is 4:
    raw = raw[8:height - 8,
          int((width - (height - 16) * (4 / 3)) / 2): width - int((width - (height - 16) * (4 / 3)) / 2)]

# 更新宽高
width = raw.shape[1]
height = raw.shape[0]

# 调用preprocess（C编译）辅助
raw = preprocess.pedestal(args["pedestal"], raw)
print(raw[0][0], raw[0][1], raw[1][0], raw[1][1])

# 分4层
plane = numpy.zeros((int(height / 2), int(width / 2), 4))
plane[:, :, 0] = raw[::2, ::2]
plane[:, :, 1] = raw[::2, 1::2]
plane[:, :, 2] = raw[1::2, ::2]
plane[:, :, 3] = raw[1::2, 1::2]

print(plane[0][0][0], plane[0][1][0], plane[1][0][0], plane[1][1][0])
print(plane[0][0][1], plane[0][1][1], plane[1][0][1], plane[1][1][1])
print(plane[0][0][2], plane[0][1][2], plane[1][0][2], plane[1][1][2])
print(plane[0][0][3], plane[0][1][3], plane[1][0][3], plane[1][1][3])

block_size_R = 100
block_size_C = 100
center = [plane.shape[0] / 2 - 1, plane.shape[1] / 2 - 1]

center_block = plane[int(center[0] - block_size_R / 2 + 1):int(center[0] + int(block_size_R / 2 + 1)),
               int(center[1] - int(block_size_C / 2) + 1): int(center[1] + int(block_size_C / 2) + 1), :]

center_mean = numpy.mean((numpy.mean(center_block, axis=0)), axis=0)
balance = numpy.max(center_mean) / center_mean

# 调用preprocess（C编译）辅助
for i in range(4):
    plane[:, :, i] = plane[:, :, i] * balance[i]

time2 = datetime.datetime.now()

print(time2 - time1)
