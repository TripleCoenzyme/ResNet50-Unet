#coding:utf-8
import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import utils
from dataset import test_dataset as Dataset
from model import Resnet_Unet as model

#7.25定稿版本，为比赛测试优化版本


#####
#可调整的参数
test_batch_size = 8#如炸显存请调小
title = 'final_test'
test_path = './test_nii/'#存放测试数据路径，根据实际情况修改
Model_path = './7.pth'#存放权重路径，根据实际情况修改
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transform = transforms.Compose([
    transforms.ToTensor()
])
save_path = './'+title+'_result/'#保存结果路径，根据实际情况修改
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#####

file_list = os.listdir(test_path)
file_list.sort()
Model = model().to(device)
utils.path_checker(save_path)
Model.load_state_dict(torch.load(Model_path))

for name in file_list:
    if name.split('.')[-1]!='nii':
        break
    test_set = Dataset(path=test_path+name, transform=transform)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

    output = []
    for index, img in enumerate(test_loader):
        if index==len(test_loader)-1:
            sys.stdout.write("\r[{}] [Batch {}/{}]\n".format(name, index+1, len(test_loader)))
        else:
            sys.stdout.write("\r[{}] [Batch {}/{}]".format(name, index+1, len(test_loader)))
        sys.stdout.flush()
        Model.eval()
        img = img.to(device)
        with torch.no_grad():
            output.append(Model(img))
            output[index] = torch.ge(output[index], 0.5).type(dtype=torch.float32) #二值化
            output[index] = utils.post_process(output[index])#后处理，结果为uint16二值numpy数组
        
        
    sys.stdout.write("\r[{}] [Saving] \n".format(name))
    sys.stdout.flush()
    result = np.concatenate(output, axis=0).squeeze()
    utils.save(case_path=test_path, save_path=save_path, case_name=name, save_name=name, img=result)
    
