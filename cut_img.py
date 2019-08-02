import random
from skimage import io
import nibabel as nib
import numpy as np

img_path='...'                                 #nii格式图片的路径信息
save_img_path='...'                            #tif格式图片路径信息
cut_img_path='...'                              #裁剪图片存储位置

#读取nii.gz的图像信息
def read_data(path):
    image_data=nib.load(path)
    return image_data

for i in range(0,195):
    path = img_path+'/SpineSagT2Wdataset3/train/image/Case%d.nii.gz'%(i+1)
    data = read_data(path)
    p=data.shape[2]     #读取每个nii.gz文件中有多少张图片，设为p
    for j in range(0,p):
        img=io.imread(save_img_path+'/Case%d_%d.tif'%(i+1,j+1), as_gray=True)
        img_mask = io.imread(save_img_path+'/mask_case%d_%d.tif'%(i+1,j+1), as_gray=True)             #读取tif图片文件
        padimg = np.pad(img, ((256, 256), (256, 256)), 'constant', constant_values=(0, 0))
        padimg_mask = np.pad(img_mask, ((256, 256), (256, 256)), 'constant', constant_values=(0, 0))    #将读取的tif图片文件周围填0，边长为256个像素
        num1 = 0                                             #黑色像素点的值大于图片像素点数量的一半，小于像素点数量的80%的图片数量，5张
        num2=0                                               #如果黑色像素点的值小于图片像素点数量的一半，每个nii.gz文件中的第1,2，最后一张图的图片数量，30张
        num3=0                                               #如果黑色像素点的值小于图片像素点数量的一半，每个nii.gz文件中除了第1,2，最后一张图的裁剪图片的图片数量，20张
        while 1:
            x = random.randint(0, padimg.shape[0]-256)
            y = random.randint(0, padimg.shape[1]-256)
            padimg_256 = padimg[(y):(y + 256), (x):(x + 256)]
            padimg_mask_256 = padimg_mask[(y):(y + 256), (x):(x + 256)]                #在填0后的tif图片中随机采取256x256的图片
            maxmum = 0
            for a in range(0,256):
                for b in range(0, 256):
                    if padimg_256[a, b] == 0:
                        maxmum += 1                                                  #记录随机取的256x256的图片中黑色像素点的数值，记为maxmum
            if maxmum > int(256*256/2) and maxmum<int(256*256*0.8) and num1<5:
                io.imsave(cut_img_path+'/Case_edge%d_%d_%d.tif'%(i+1,j+1,num1 + 1), padimg_256)
                io.imsave(cut_img_path+'/Mask_edge%d_%d_%d.tif'%(i+1,j+1,num1 + 1), padimg_mask_256)
                num1 += 1                                                            #如果黑色像素点的值大于图片像素点数量的一半，小于像素点数量的80%，则储存这张图片
            if maxmum<int(256*256/2) and num2<30 and num3<20:
                if j+1==1 or j+1==2 or j+1==p:
                    io.imsave(cut_img_path+'/Case_norm%d_%d_%d.tif' % (i + 1, j+1, num2 + 1), padimg_256)
                    io.imsave(cut_img_path+'/Mask_norm%d_%d_%d.tif' % (i + 1, j+1, num2 + 1),padimg_mask_256)
                    num2=num2+1                                                      #如果黑色像素点的值小于图片像素点数量的一半，每个nii.gz文件中的第1,2，最后一张图的裁剪图片储存下来
                else:
                    io.imsave(cut_img_path+'/Case_norm%d_%d_%d.tif' % (i + 1, j + 1, num3 + 1),padimg_256)
                    io.imsave(cut_img_path+'/Mask_norm%d_%d_%d.tif' % (i + 1, j + 1, num3 + 1),padimg_mask_256)
                    num3=num3+1                                                      #如果黑色像素点的值小于图片像素点数量的一半，每个nii.gz文件中除了第1,2，最后一张图的裁剪图片储存下来
            if num1 == 5 and (num2==30 or num3==20):
                break