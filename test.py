import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import utils
from dataset import Dataset
from model import Resnet_Unet as model


#####
#可调整的参数
test_batch_size = 1
title = 'ResNet_final'
path = '../'
test_path = path+'imgs/test/'
Model_path = '../7.pth'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
transform = transforms.Compose([
    transforms.ToTensor()
])
save_path = path+'log/'+title+'_test/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#####


test_set = Dataset(path=test_path, transform=transform, mode='test')
test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)

Model = model(BN_enable=True, resnet_pretrain=False).to(device)
utils.path_checker(save_path)

Model.load_state_dict(torch.load(Model_path))
Model.eval()
for index, (img,name) in enumerate(test_loader):
    Model.eval()
    img = img.to(device)
     
    with torch.no_grad():
        output = Model(img)
        output = torch.ge(output, 0.5).type(dtype=torch.float32) #二值化
        output = utils.post_process(output)#后处理
        for i in range(test_batch_size):
            vutils.save_image(output[i,:,:,:], save_path+name[i].split('/')[1], padding=0)
    sys.stdout.write("\r[test] [Epoch {}/{}] [Batch {}/{}]".format(7, 10, index+1, len(test_loader)))
    sys.stdout.flush()
