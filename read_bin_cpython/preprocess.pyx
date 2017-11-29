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
            # threshold the pixel
            raw_image[y, x] += value

    # return image
    return raw_image
