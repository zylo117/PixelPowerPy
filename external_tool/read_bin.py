import array
import datetime
import numpy as np


class BinFile:
    def __init__(self, filepath, datatype=np.uint16):
        self.filepath = filepath
        # self.rawdata = array.array("H", open(self.filepath, "rb").read())
        self.rawdata = np.fromfile(filepath, dtype=np.uint16)  # 75% faster than original python embedded method, power of philosophy!
        self.header = None
        self.realdata = None

    def get_realdata(self, header_length):
        self.header = self.rawdata[:header_length]
        self.realdata = self.rawdata[header_length:]
        return self.header, self.realdata


if __name__ == "__main__":
    time1 = datetime.datetime.now()

    bin = BinFile("/Volumes/OSX_Data/Github/PixelPowerPy/io_bin/b.bin")
    header, realdata = bin.get_realdata(2)

    time2 = datetime.datetime.now()
    print(time2 - time1)
    print(0)
