import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("123.png", 0)  # 直接读为灰度图像
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
# 取绝对值：将复数变化成实数
# 取对数的目的为了将数据变化到0-255
s1 = np.log(np.abs(fshift))
plt.subplot(221), plt.imshow(img, 'gray'), plt.title('original')
plt.xticks([]), plt.yticks([])
# ---------------------------------------------
# 逆变换--取绝对值就是振幅
f1shift = np.fft.ifftshift(np.abs(fshift))
img_back = np.fft.ifft2(f1shift)
# 出来的是复数，无法显示
img_back = np.abs(img_back)
# 调整大小范围便于显示
img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
plt.subplot(222), plt.imshow(img_back, 'gray'), plt.title('only Amplitude')
plt.xticks([]), plt.yticks([])
# ---------------------------------------------
# 逆变换--取相位
f2shift = np.fft.ifftshift(np.angle(fshift))
img_back = np.fft.ifft2(f2shift)
# 出来的是复数，无法显示
img_back = np.abs(img_back)
# 调整大小范围便于显示
img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
plt.subplot(223), plt.imshow(img_back, 'gray'), plt.title('only phase')
plt.xticks([]), plt.yticks([])
# ---------------------------------------------
# 逆变换--将两者合成看看
s1 = np.abs(fshift)  # 取振幅
s1_angle = np.angle(fshift)  # 取相位
s1_real = s1 * np.cos(s1_angle)  # 取实部
s1_imag = s1 * np.sin(s1_angle)  # 取虚部
s2 = np.zeros(img.shape, dtype=complex)
s2.real = np.array(s1_real)  # 重新赋值给s2
s2.imag = np.array(s1_imag)

f2shift = np.fft.ifftshift(s2)  # 对新的进行逆变换
img_back = np.fft.ifft2(f2shift)
# 出来的是复数，无法显示
img_back = np.abs(img_back)
# 调整大小范围便于显示
img_back = (img_back - np.amin(img_back)) / (np.amax(img_back) - np.amin(img_back))
plt.subplot(224), plt.imshow(img_back, 'gray'), plt.title('another way')
plt.xticks([]), plt.yticks([])
plt.show()
