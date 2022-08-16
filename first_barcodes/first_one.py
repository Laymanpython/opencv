import pylab  # plt不显示以后添加pylab包进行显示
import cv2  # cv2进行读取图片以及进行相关操作
import matplotlib.pyplot as plt  # 绘制图片
import numpy as np

img = cv2.imread(r"barcodes.jpg", cv2.IMREAD_GRAYSCALE)  # 以灰度图形式读取图片
img_out = cv2.imread(r"barcodes.jpg")
# print(img.shape)  # 二维形式即单通道
# plt.imshow(img, cmap='Greys_r')  # 显示图片
# pylab.show()  # 显示图片

# 图像缩放
scale = 800.0 / img.shape[1]
img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

# 黑帽运算
kernel = np.ones((1, 3), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, anchor=(1, 0))

# 二值化处理
thresh, img = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
# plt.imshow(img, cmap='Greys_r')  # 显示图片
# pylab.show()  # 显示图片

kernel = np.ones((1, 5), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel, anchor=(2, 0), iterations=1)
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, anchor=(2, 0), iterations=1)

kernel = np.ones((21, 35), np.uint8)
img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=1)
#plt.imshow(img, cmap='Greys_r')  # 显示图片
#pylab.show()  # 显示图片

contours, hierarachy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

unscale = 1.0 / scale
if contours != None:
    for contour in contours:

        if cv2.contourArea(contour)<=2000:
            continue;

        rect = cv2.minAreaRect(contour)

        rect = \
            ((int(rect[0][0]*unscale),int(rect[0][1]*unscale)), \
             (int(rect[1][0] * unscale), int(rect[1][1] * unscale)), \
             rect[2])

        box = np.int64(cv2.boxPoints(rect))
        cv2.drawContours(img_out,[box],0,(0,255,0),thickness=2)


plt.imshow(img_out)
pylab.show()
cv2.imwrite("out.png",img_out)