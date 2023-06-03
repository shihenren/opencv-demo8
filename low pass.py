import numpy as np
import cv2
import matplotlib.pyplot as plt

#展示图像的方法
def cv_show(title,img):
    cv2.imshow(title,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

img = cv2.imread('bird.jpg',0)#读取灰度图
cv_show('img',img)#展示一下图像，按任意键继续

img_float = np.float32(img)#处理前先把图像的dtype转换成float32，原本是uint8,傅里叶变换得用float32类型才能计算

#傅里叶变换
dft = cv2.dft(img_float,flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

rows,cols = img.shape#分别得到图像的长和宽
crows, ccol =int(rows/2),int(cols/2)#分别得到图像中心点的位置

#低通滤波，创建低通滤波器掩码，中心区域是1，周围区域是0
mask = np.ones((rows, cols, 2), dtype = np.uint8)#2是因为傅里叶变换结果是双通道的，分别为实部和虚部
mask[crows-30:crows+30,ccol-30:ccol+30] = 0

#IDFT,进行一个逆变换
fshift = dft_shift*mask#mask是中间是1，周围是0的掩膜，我们之前通过 np.fit.fftshift将低频部分移到图像中间了，乘以mask这样就能将低频部分保留（*1），高频部分抑制（*0）
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)#这两步就是之前的傅里叶变换部分的逆变换
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray'),plt.title('origin')
plt.subplot(1,2,2),plt.imshow(img_back,cmap = 'gray'),plt.title('high-pass filter')
plt.show()
