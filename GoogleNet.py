import torch
import  torch.nn as nn
import  torchvision
from torch.utils.data import  DataLoader
class GoogleNet(nn.Module):
    def __init__(self,in_c=3,class_n=1000):
        super(GoogleNet, self).__init__()
        self.conv1=Conv_Block(input_channel=in_c,output_chanel=64,kernel_size=7,stride=2,padding=3)
        self.maxpool1 = nn.MaxPool2d(stride=2, kernel_size=3, padding=1)
        self.conv2=Conv_Block(input_channel=64,output_chanel=192,kernel_size=3,stride=1,padding=1)
        self.maxpool2 = nn.MaxPool2d(stride=2, kernel_size=3, padding=1)
        self.inception3a=Inception_Block(in_c=192,out_1x1=64,r3_1x1=96,o3_1x1=128,r5_1x1=16,o5_1x1=32,out_pool_1x1=32)
        self.inception3b=Inception_Block(in_c=256,out_1x1=128,r3_1x1=128,o3_1x1=192,r5_1x1=32,o5_1x1=96,out_pool_1x1=64)
        self.maxpool3=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)

        self.inception4a=Inception_Block(in_c=480,out_1x1=192,r3_1x1=96,o3_1x1=208,r5_1x1=16,o5_1x1=48,out_pool_1x1=64)
        self.inception4b=Inception_Block(in_c=512,out_1x1=160,r3_1x1=112,o3_1x1=224,r5_1x1=24,o5_1x1=64,out_pool_1x1=64)
        self.inception4c=Inception_Block(in_c=512,out_1x1=128,r3_1x1=128,o3_1x1=256,r5_1x1=24,o5_1x1=64,out_pool_1x1=64)
        self.inception4d=Inception_Block(in_c=512,out_1x1=112,r3_1x1=144,o3_1x1=288,r5_1x1=32,o5_1x1=64,out_pool_1x1=64)
        self.inception4e=Inception_Block(in_c=528,out_1x1=256,r3_1x1=160,o3_1x1=320,r5_1x1=32,o5_1x1=128,out_pool_1x1=128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2,padding=1)
        self.inception5a = Inception_Block(in_c=832, out_1x1=256, r3_1x1=160, o3_1x1=320, r5_1x1=32, o5_1x1=128,
                                           out_pool_1x1=128)
        self.inception5b = Inception_Block(in_c=832, out_1x1=384, r3_1x1=192, o3_1x1=384, r5_1x1=48, o5_1x1=128,
                                           out_pool_1x1=128)
        self.ave_pool=nn.AvgPool2d(kernel_size=7,stride=1)
        self.dropout=nn.Dropout(p=0.4)
        self.fc=nn.Linear(1024,class_n)
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool1(x)
        x=self.conv2(x)
        x=self.maxpool2(x)
        x=self.inception3a(x)
        x=self.inception3b(x)
        x=self.maxpool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x=self.maxpool4(x)
        x=self.inception5a(x)
        x=self.inception5b(x)
        x=self.ave_pool(x)
        x=x.reshape(x.shape[0],-1)

        x=self.dropout(x)
        x=self.fc(x)
        return  x

class Conv_Block(nn.Module):
    def __init__(self,input_channel,output_chanel,**kwargs):
        super(Conv_Block, self).__init__()
        self.conv=nn.Conv2d(in_channels=input_channel,out_channels=output_chanel,**kwargs)
        self.bn=nn.BatchNorm2d(output_chanel)
        self.relu=nn.ReLU()
    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))
class Inception_Block(nn.Module):
    def __init__(self,in_c,out_1x1,r3_1x1,o3_1x1,r5_1x1,o5_1x1,out_pool_1x1):
        super(Inception_Block, self).__init__()
        self.branch1=Conv_Block(input_channel=in_c,output_chanel=out_1x1,kernel_size=1,stride=1)
        self.branch2=nn.Sequential(Conv_Block(input_channel=in_c,output_chanel=r3_1x1,kernel_size=1,stride=1),
                                    Conv_Block(input_channel=r3_1x1,output_chanel=o3_1x1,kernel_size=3,stride=1,padding=1))
        self.branch3=nn.Sequential(Conv_Block(input_channel=in_c,output_chanel=r5_1x1,kernel_size=1,stride=1),
                                    Conv_Block(input_channel=r5_1x1,output_chanel=o5_1x1,kernel_size=5,stride=1,padding=2))

        self.branch4=nn.Sequential(nn.MaxPool2d(stride=1, kernel_size=3, padding=1),
                                   Conv_Block(input_channel=in_c,output_chanel=out_pool_1x1,kernel_size=1,stride=1))
    def forward(self,x):
        x1=self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        return torch.cat([self.branch1(x),self.branch2(x),self.branch3(x),self.branch4(x)],1)


if __name__=='__main__':
    x=torch.rand(17,3, 224,224)
    model=GoogleNet()
    out=model(x)
    print(out.shape)




