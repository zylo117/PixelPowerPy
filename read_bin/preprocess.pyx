# 图片增益
cpdef unsigned short[:, :] pedestal(int value, unsigned short[:, :] raw_image):
    # set the variable extension types
    cdef int x, y, w, h

    # grab the image dimensions
    h = raw_image.shape[0]
    w = raw_image.shape[1]

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            # 增益
            raw_image[y, x] += value

    # return image
    return raw_image

# 图片白平衡辅助
cpdef unsigned short[:, :] whitebalance(unsigned short[:, :] raw_image):
    # set the variable extension types
    cdef int x, y, w, h

    # grab the image dimensions
    h = raw_image.shape[0]
    w = raw_image.shape[1]

    # loop over the image
    for y in range(0, h):
        for x in range(0, w):
            # 遍历，每个像素乘平衡因子
            raw_image[y, x] += 1

    # return image
    return raw_image
