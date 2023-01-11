import torch
import torch.nn as nn
class block(nn.Module):
    def __init__(self,in_chanel,out_chanel,stride=1,down_sample=None):
        super(block, self).__init__()
        self.expansion=4
        self.down_sample=down_sample
        self.conv1=nn.Conv2d(in_channels=in_chanel,out_channels=out_chanel,stride=stride,kernel_size=1)
        self.BN1 = nn.BatchNorm2d(out_chanel)
        self.conv2 = nn.Conv2d(in_channels=out_chanel, out_channels=out_chanel, stride=1, padding=1,kernel_size=3)
        self.BN2 = nn.BatchNorm2d(out_chanel)
        self.conv3=nn.Conv2d(in_channels=out_chanel, out_channels=self.expansion*out_chanel, stride=1,kernel_size=1)
        self.BN3=nn.BatchNorm2d(self.expansion*out_chanel)
        self.relu=nn.ReLU()
    def forward(self,x):
        idetity=x
        x=self.conv1(x)
        x=self.BN1(x)
        x=self.relu(x)
        x=self.conv2(x)
        x=self.BN2(x)
        x=self.relu(x)
        x=self.conv3(x)
        x=self.BN3(x)
        if self.down_sample is not None:
            idetity=self.down_sample(idetity)
        x+=idetity
        x=self.relu(x)
        return x
class ResNet(nn.Module):
    def __init__(self,layer_num):
        super(ResNet, self).__init__()
        self.in_channel=64
        self.conv1=nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2,padding=3)
        self.maxpool=nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        self.layer_1=self._make_layer(layer_num[0],64,stride=1)
        self.layer_2 = self._make_layer(layer_num[1], 128, stride=2)
        self.layer_3 = self._make_layer(layer_num[2], 256, stride=2)
        self.layer_4 = self._make_layer(layer_num[3], 512, stride=2)
        self.ave_pool=nn.AdaptiveAvgPool2d((1,1))
        self.fc=nn.Linear(512*4, 1000)
    def forward(self,x):
        x=self.conv1(x)
        x=self.maxpool(x)
        # print(self.layer_1)
        x=self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x=self.ave_pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc(x)
        return x
    def _make_layer(self,layer_num,out_chanel,stride):
        layer=[]
        down_sample=None
        if stride!=1 or self.in_channel!=4*out_chanel:# first convolution in each block, it is stride 2 or the size of input channel is not equal to out put channel
            down_sample=nn.Sequential(nn.Conv2d(in_channels=self.in_channel,out_channels=4*out_chanel,kernel_size=1,stride=stride),
                                      nn.BatchNorm2d(4*out_chanel))
        layer.append(block(self.in_channel,out_chanel,stride=stride,down_sample=down_sample))
        self.in_channel=4*out_chanel
        for i in range(layer_num-1):
            layer.append(block(self.in_channel,out_chanel))
        return nn.Sequential(*layer)


image=torch.randn(6,3,224,224)
list_=[3,8,36,3] # 50, 131, 101,.... is defined by this list here
model=ResNet(list_)
x=model(image)

print(x.shape)

