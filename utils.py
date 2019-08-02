import os
import random
import sys
import torch
from skimage.morphology import remove_small_objects,binary_opening
import numpy as np
import nibabel as nib

#7.25定稿版本
#经过比赛测试优化，尤其是post_process函数，输出有较大变化

def path_checker(path):
    """
    检查目录是否存在，不存在，则创建
    """
    if not os.path.isdir(path):
        os.makedirs(path)
        print(path+'不存在，已创建...')
    else:
        print(path+'已存在')


def get_val_case(path, val_num = 15):
    """
    从该地址文件夹下case中随机选取val_num个case作为val
    """
    case_pool=list(range(1,196))
    if not os.path.isdir(path+'case/'):
        print ('case error')
    else:
        if not os.path.isfile(path+'val_case.txt'):
            random.shuffle(case_pool)
            with open(path+'val_case.txt','w') as f1:
                    for name in case_pool[:val_num]:
                        f1.write(str(name)+'\n')


def get_img_list(path, save_mode, val_path = None):
    """
    遍历该地址文件夹下case中的所有图片文件，将相对路径写入同一位置给定文件名的文件
    save_mode为'train''val'与'test'
    'train' mode需要先进行get_val_case，将val以外数据文件列表打乱存入train.txt
    'val' mode需要先进行get_val_case，根据val.txt读取所有的val图片
    'test' mode将数据文件列表存入test.txt
    val_path表示整图储存位置
    """
    path_pool=[]
    get_val_case(path)
    if not os.path.isdir(path+'case/'):
        print ('case error')
    else:
        if save_mode == 'train':
            for root, _, files in os.walk(path+'case/'):
                if len(files)>0:
                    for name in files:
                        tname = root.split(path+'case/')[-1]
                        path_pool.append(tname+'/'+name)
            random.shuffle(path_pool)
            with open(path+'val_case.txt','r') as f1:
                val_pool=f1.readlines()
                for i in range(len(val_pool)):
                    val_pool[i]=val_pool[i].split('\n')[0]
            with open(path+'train.txt','w') as f2:
                for name in path_pool:
                    if not(name.split('/')[1].split('_')[0] in val_pool):
                        f2.write(name+'\n')
        
        elif save_mode == 'val':
            with open(path+'val_case.txt','r') as f1:
                val_pool=f1.readlines()
                for i in range(len(val_pool)):
                    val_pool[i]='case'+val_pool[i].split('\n')[0]
            for root, _, files in os.walk(val_path+'case/'):
                if len(files)>0:
                    for name in files:
                        tname = root.split(val_path+'case/')[-1]
                        path_pool.append(tname+'/'+name)
            with open(val_path+'val.txt','w') as f2:
                for name in path_pool:
                    if name.split('/')[0] in val_pool:
                        f2.write(name+'\n')
            
        elif save_mode == 'test':
            with open(path+'test.txt','w') as f:
                for name in path_pool:
                    f.write(name+'\n')

def average(list):
    s = 0
    for item in list:
        s += item
    return s/len(list)

def sum(list):
    s = 0
    for item in list:
        s += item
    return s

def analysis(x,y):
    '''
    对输入的两个四维张量[B,1,H,W]进行逐图的DSC、PPV、Sensitivity计算
    其中x表示网络输出的预测值
    y表示实际的预想结果mask
    返回为一个batch中DSC、PPV、Sen的平均值及batch大小
    '''
    x=x.type(dtype=torch.uint8)
    y=y.type(dtype=torch.uint8)#保证类型为uint8
    DSC=[]
    PPV=[]
    Sen=[]
    if x.shape==y.shape:
        batch=x.shape[0]
        for i in range(batch):#按第一个维度分开
            
            tmp = torch.eq(x[i],y[i])
            
            tp=int(torch.sum(torch.mul(x[i]==1,tmp==1))) #真阳性
            fp=int(torch.sum(torch.mul(x[i]==1,tmp==0))) #假阳性
            fn=int(torch.sum(torch.mul(x[i]==0,tmp==0))) #假阴性
        
        
            try:
                DSC.append(2*tp/(fp+2*tp+fn))
            except:
                DSC.append(0)
            try:
                PPV.append(tp/(tp+fp))
            except:
                PPV.append(0)
            try:
                Sen.append(tp/(tp+fn))
            except:
                Sen.append(0)
            
                
    else:
        sys.stderr.write('Analysis input dimension error')
        

    DSC = sum(DSC)/batch
    PPV = sum(PPV)/batch
    Sen = sum(Sen)/batch
    return DSC, PPV, Sen, batch


def post_process(img,min_size=100):
    '''
    图像后处理过程
    包括开运算和去除过小体素
    返回uint16格式numpy二值数组
    '''
    img = img.cpu()
    img = img.numpy().astype(np.bool)
    b,c,w,h = img.shape
    if c==1:
        for i in range(b):
            img_tmp = img[i,0,:,:]
            img_tmp = binary_opening(img_tmp)
            remove_small_objects(img_tmp, min_size=min_size, in_place=True)
            img_tmp = ~remove_small_objects(~img_tmp, min_size=min_size)
            img[i,0,:,:] = img_tmp
        
    return img.astype(np.uint16)

def get_array(path):
    '''
    从nii文件中获取三维uint8 numpy数组[c,w,h]
    '''
    image_data1 = nib.load(path).get_data()  # 读取nii的图像信息
    image_data2 = nib.load(path)
    cmin = image_data1[:, :, 0].min()
    cmax = image_data1[:, :, 0].max()
    high = 255
    low = 0
    cscale = cmax - cmin
    scale = float(high - low) / cscale
    bytedata = (image_data1[:, :, 0] - cmin) * scale + low
    image = (bytedata.clip(low, high) + 0.5).astype(np.uint8)
    for j in range(1, image_data2.shape[2]):
        cmin = image_data1[:, :, j].min()
        cmax = image_data1[:, :, j].max()
        high = 255
        low = 0
        cscale = cmax - cmin
        scale = float(high - low) / cscale
        bytedata = (image_data1[:, :, j] - cmin) * scale + low
        imagex = (bytedata.clip(low, high) + 0.5).astype(np.uint8)
        image = np.append(image, imagex)
    img_f = image.reshape(image_data2.shape[2], image_data2.shape[0], image_data2.shape[1])
    return img_f


def save(case_path , save_path, case_name, save_name, img):
    case_data = nib.load(case_path+case_name)
    x = img.transpose(1, 2, 0)
    x = nib.Nifti1Image(x, case_data.affine)
    nib.save(x, save_path+save_name)
