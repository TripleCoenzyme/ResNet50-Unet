import nibabel as nib
import scipy.misc

img_path='...'                                 #nii格式图片的路径信息
save_img_path='...'                            #tif格式图片路径信息

#读取nii.gz的图像信息和形状信息
def read_data(path):
    image_data1 = nib.load(path).get_data()     #读取nii.gz的图像信息
    image_data2=nib.load(path)                  #读取nii.gz的形状信息
    return image_data1,image_data2

#读取trian/image文件夹中的图片
for j in range(0,195):
    path = img_path+'/SpineSagT2Wdataset3/train/image/Case%d.nii.gz'%(j+1)
    data1,data2 = read_data(path)
    for i in range(0,data2.shape[2]):
        scipy.misc.imsave(save_img_path+'/Case%d_%d.tif'%(j+1,i+1), data1[:,:,i])#存储为tif格式的图片

#读取trian/groundtruth文件夹中的图片
for j in range(0,195):
    path =img_path+ '/SpineSagT2Wdataset3/train/groundtruth/mask_case%d.nii.gz'%(j+1)
    data1, data2 = read_data(path)
    for i in range(0,data2.shape[2]):
        scipy.misc.imsave(save_img_path+'/mask_case%d_%d.tif'%(j+1,i+1), data1[:,:,i])#存储为tif格式的图片

#读取test/image文件夹中的图片
for j in range(195,210):
    path = img_path+'/SpineSagT2Wdataset3/test/image/Case%d.nii.gz'%(j+1)
    data1, data2 = read_data(path)
    for i in range(0,data2.shape[2]):
        scipy.misc.imsave(save_img_path+'/Case%d_%d.tif'%(j+1,i+1), data1[:,:,i])     #存储为tif格式的图片