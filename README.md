# ResNet50-Unet

本项目获第五届全国大学生生物医学工程创新设计竞赛命题组一等奖。

This project is rewarded for 2019 National Undergraduate Biomedical Engineering Inovation Design Competition.

## Introduction

The U-Net uses the first 4 layers of ResNet50 for the downsampling part and replace the transposed convolution with Pixel Shuffle in the upsampling part.
References can be found in model.py.

***[Update] This project is based on pytorch 1.0 and may contain some deprecated code. For example, the different position requirement of pytorch scheduler and the native support of tensorboard. To get a modern version of this project, I suggest to have a look at the [MRI_MotionCorrection](https://github.com/TripleCoenzyme/MRI_MotionCorrection) project.***

## Simple Manual

read_img.py用于将数据集中的nii.gz文件存储为tif图片格式

read_img.py is used to save the .nii.gz files into .tif pictures.

cut_img.py用于裁剪tif格式图片生成训练集

cut_img.py is used to crop the .tif pictures for training dataset.

save_img.py用于将测试集的结果存储为nii.gz文件格式

save_img.py is used to save the test result back to .nii.gz files.

utils.py中包含网络训练、测试中所需要的一些辅助函数

utils.py contains some supporting functions for model training and testing.

model.py为网络结构程序

model.py contains the ResNet-UNet model.

dataset.py为数据集构建程序

dataset.py contains the dataset code for pytorch.

main.py为网络训练用程序，验证过程也包含在其中

main.py is the training code including validation.

test.py为网络测试用程序

test.py is the testing code.

final_test.py为比赛现场测试脚本，从给定的nii文件直接生成结果nii文件

final_test.py is the testing code for live testing, including slicing from input .nii file and rebuilding into .nii file.

