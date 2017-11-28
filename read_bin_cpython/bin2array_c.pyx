def b2a(raw_image_data):
    raw = open(raw_image_data, "rb").read()

    raw_dec = [];

    for i in range(int(len(raw) / 2)):
        raw_dec.append(int(str(hex(raw[2 * i + 1]))[2:] + str(hex(raw[2 * i]))[2:], 16))

    return raw_dec
