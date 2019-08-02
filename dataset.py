import os
import torch
from PIL import Image
import utils
from torchvision import transforms
from torch import nn
import numpy as np

#7.25定稿版本，用于比赛测试

class Dataset(torch.utils.data.Dataset):
    '''
    给定三种取数据mode：'train'/'test'/'validation'
    '''
    def __init__(self, path, transform=None, mode='train', val_path = None):
        self.img_path = path+'case/'
        self.mask_path = path+'mask/'
        self.transform = transform
        self.mode = mode
        if self.mode == 'train':
            if not(os.path.isfile(path+'train.txt')):
                utils.get_img_list(path, 'train')
            with open(path+'train.txt','r') as f_train:
                log = f_train.readlines()
        elif self.mode == 'validation':
            self.val_path=val_path
            if not(os.path.isfile(path+'val_case.txt')):
                utils.get_img_list(path, 'val', val_path=val_path)
            with open(val_path+'val.txt','r') as f_val:
                log = f_val.readlines()
        elif self.mode == 'test':
            if not(os.path.isfile(path+'test.txt')):
                utils.get_img_list(path, 'test')
            with open(path+'test.txt','r') as f_test:
                log = f_test.readlines()
        
        self.log = log

    def __getitem__(self,index):
        if self.mode == 'train' :
            img = Image.open(self.img_path+self.log[index].strip())
            mask = Image.open(self.mask_path+self.log[index].strip())
            if self.transform is not None:
                img = self.transform(img)
                mask = self.transform(mask)
            return [img, mask]
        elif self.mode == 'validation':
            pad = []
            img = Image.open(self.val_path+'case/'+self.log[index].strip())
            mask = Image.open(self.val_path+'mask/'+self.log[index].strip())
            if self.transform is not None:
                img = self.transform(img)
                mask = self.transform(mask)

            return [img, mask]
        elif self.mode == 'test':
            img = Image.open(self.img_path+self.log[index].strip())
            if self.transform is not None:
                img = self.transform(img)
            return [img,self.log[index].strip()]

    def __len__(self):
        return len(self.log)




class test_dataset(torch.utils.data.Dataset):
    '''
    比赛测试专用dataset
    给定一个nii文件地址，返回该文件的所有切片
    '''
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform
        self.array = utils.get_array(path)

    def __getitem__(self,index):
        img = self.array[index,:,:]
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return np.shape(self.array)[0]