def b2a(raw_image_data):
    raw = open(raw_image_data, "rb").read()

    raw_dec = [];
    print(len(raw))
    for i in range(int(len(raw) / 2)):
        pre = hex(raw[2 * i + 1])[2:]
        post = hex(raw[2 * i])[2:]
        if len(pre) is 1:
            pre = "0" + pre
        if len(post) is 1:
            post = "0" + post
        raw_dec.append(int(str(pre) + str(post), 16))

    return raw_dec
