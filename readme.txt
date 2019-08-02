read_img.py用于将数据集中的nii.gz文件存储为tif图片格式
cut_img.py用于裁剪tif格式图片生成训练集
save_img.py用于将测试集的结果存储为nii.gz文件格式
utils.py中包含网络训练、测试中所需要的一些辅助函数
model.py为网络结构程序
dataset.py为数据集构建程序
main.py为网络训练用程序，验证过程也包含在其中
test.py为网络测试用程序
final_test.py为比赛现场测试脚本，从给定的nii文件直接生成结果nii文件



The u-net uses the first 4 layers of ResNet50 for the downsampling part and replace the transposed convolution with Pixel Shuffle in the upsampling part.
