import os
import sys
import time

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import utils as vutils

import utils
from dataset import Dataset
from model import Resnet_Unet as model

#6.9定稿版本


###########
#可调整的训练超参数
batch_size = 16
val_batch_size = 2
lr =1e-3
start_epoch = 0
stop_epoch = 10
###########


###########
#可调整的路径参数
title = 'ResNet_final'
path = '/mnt/diskarray/fj/ResNet-Unet/'
data_path = path+'cut_imgs/'
val_path = path+'imgs/train/'
Model_path = path+'log/checkpoints/'+title+'/4.pth'
###########


###########
#可调整的训练相关处理
pretrain = True
multi_GPU = False
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
save_step = 10 #决定多少次保存一次可视化结果
transform = transforms.Compose([
    #transforms.Resize((256,256)),
    transforms.ToTensor()
])
###########


###########
#无需调整的路径参数
log_path = path+'log/'
checkpoints_path = path+'log/checkpoints/'+title+'/'
tensorboard_path = path+'log/tensorboard/'+title+'/'
visualize_path = path+'log/visualize/'+title+'/'
###########



utils.path_checker(log_path)
utils.path_checker(checkpoints_path)
utils.path_checker(tensorboard_path)
utils.path_checker(visualize_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Writer = SummaryWriter(tensorboard_path)

train_set = Dataset(path=data_path, transform=transform, mode='train')
val_set = Dataset(path=data_path, transform=transform, mode='validation', val_path=val_path)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=val_batch_size, shuffle=False)

Model = model(BN_enable=True, resnet_pretrain=False).to(device)
if pretrain:
    Model.load_state_dict(torch.load(Model_path))


criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(Model.parameters(),lr=lr)
#optimizer = torch.optim.SGD(Model.parameters(),lr=lr,momentum=0.9,weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)



###########
#开始训练

for epoch in range(start_epoch, stop_epoch):
    scheduler.step()
    batch_sum = len(train_loader)
    #训练部分
    for index, (img,mask) in enumerate(train_loader):
        img = img.to(device)
        mask = mask.to(device)
        
        Model.train()
        Model.zero_grad()
        output = Model(img)

        loss = criterion(output, mask)
        loss.backward()
        optimizer.step()
        if index%save_step==0:
            
            output_img = vutils.make_grid(output[0,:,:,:], padding=0, normalize=True, range=(0,255))
            output_tmp = torch.ge(output, 0.5).mul(255)
            Writer.add_scalar('scalar/loss', loss, index)
            Writer.add_image('image/input', img[0,:,:,:])
            Writer.add_image('image/mask', mask[0,:,:,:])
            Writer.add_image('image/predict', output_tmp[0,:,:,:])
            Writer.add_image('image/output', output[0,:,:,:])
            
        sys.stdout.write("\r[Train] [Epoch {}/{}] [Batch {}/{}] [loss:{:.8f}] [learning rate:{}]".format(epoch+1, stop_epoch, index+1, batch_sum, loss.item(), optimizer.param_groups[0]['lr']))
        sys.stdout.flush()

    #保存权重，每个epoch进行一次保存
    
    torch.save(Model.state_dict(), checkpoints_path+'{}.pth'.format(epoch+1))

    #验证部分
    DSC_sum = 0
    PPV_sum = 0
    Sen_sum = 0
    batch_sum = 0
    for index, (img,mask) in enumerate(val_loader):
        Model.eval()
        DSC = 0
        PPV = 0
        Sen = 0
        batch = 0
        img = img.to(device)
        mask = mask.to(device)
        
        with torch.no_grad():

            output = Model(img)           
            output = torch.ge(output, 0.5).type(dtype=torch.float32) #二值化
            output = utils.post_process(output) #后处理

            DSC ,PPV, Sen, batch = utils.analysis(output,mask)
            DSC_sum += DSC*batch
            PPV_sum += PPV*batch
            Sen_sum += Sen*batch
            batch_sum += batch
            if index%save_step==0:
                img_list = [
                    img[0,:,:,:],
                    output[0,:,:,:],
                    mask[0,:,:,:]
                ]
                img_visualize = vutils.make_grid(img_list)
                visualize_img_path = visualize_path+str(epoch)+'_'+str(index+1)+'.tif'
                vutils.save_image(img_visualize, visualize_img_path)
        sys.stdout.write("\r[Val] [Epoch {}/{}] [Batch {}/{}] [DSC:{:.5f}] [PPV:{:.5f}] [Sen:{:.5f}]".format(epoch+1, stop_epoch, index+1, len(val_loader), DSC, PPV, Sen))
        sys.stdout.flush()

    DSC_sum /= batch_sum
    PPV_sum /= batch_sum
    Sen_sum /= batch_sum

    with open(log_path+title+'.txt','a') as f:
        f.write('{}\t{:.5f}\t{:.5f}\t{:.5f}\n'.format(epoch+1, DSC_sum, PPV_sum, Sen_sum))

        
    
