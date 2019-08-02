import nibabel as nib
from skimage import io
import numpy as np

#将tif整合为nii.gz的格式，path1为tif图的路径，path2为存储nii图的路径，path3为原nii格式图片的路径信息，d为整合图片的编号
def save(path1,path2,path3,d):
    image_data = nib.load(path3+'/SpineSagT2Wdataset3/test/image/Case%d.nii.gz'%d)  #原nii格式图片的图片信息
    p=image_data.shape[2]                               #读取tif图的张数，记为p
    img_maskx = io.imread(path1+'/mask_case%d_1.tif'%d, as_gray=True)
    data1 = img_maskx.shape                             #读取tif图的形状
    for j in range(1, p):
        img_mask = io.imread(path1+'/mask_case%d_%d.tif' % (d,j + 1), as_gray=True)
        img_maskx = np.append(img_maskx, img_mask)          #将所有tif图片整合为一个一维数组
    img_maskf = img_maskx.reshape(p, data1[0], data1[1])    #将一维数组整合为3维数组
    x = img_maskf.transpose(1, 2, 0)                       #变换三维数组坐标轴
    x = x.astype(np.short)                                #改变数组类型
    array_img = nib.Nifti1Image(x, image_data.affine)            #用原nii文件的映射矩阵做三维数组的变换
    nib.save(array_img,path2+ '/mask_case%d.nii.gz'%d)

tif_img_path='...'          #得到tif图结果的存储路径
nii_img_path='...'          #整合为nii图的存储路径
img_path='...'              #原nii格式图片的路径信息
for i in range(196,211):
    save(tif_img_path,nii_img_path,img_path,i)



