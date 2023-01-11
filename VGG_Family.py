import torch
import torchvision
import torch.nn as nn
vgg={'VGG11':[64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
     'VGG13':[64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
     'VGG16':[64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
     'VGG19':[64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']}
class VGG(nn.Module):
    def __init__(self,input_chennel=3,num_class=1000):
        super(VGG, self).__init__()
        self.input_channel=input_chennel
        self.num_class=num_class
        self.conv_layer=self.layer(vgg['VGG19'])
        self.fc=nn.Sequential(nn.Linear(7*7*512,4096),nn.ReLU(),
                              nn.Dropout(p=0.5),nn.Linear(4096,4096),nn.ReLU(),
                              nn.Dropout(p=0.5),nn.Linear(4096,self.num_class))
    def layer(self,architecture):
            layer_ope=[]
            x=self.input_channel
            for layer_ in architecture:
                if type(layer_)==int:
                    layer_ope+=[nn.Conv2d(in_channels=x,out_channels=layer_,
                                          kernel_size=(3,3),stride=(1,1),padding=(1,1)),nn.ReLU(),nn.BatchNorm2d(layer_)]
                    x = layer_

                else:
                    layer_ope+=[nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))]
            return nn.Sequential(*layer_ope)


    def forward(self,x):
            x=self.conv_layer(x)
            x=x.reshape(x.shape[0],-1)
            x=self.fc(x)
            return x
model=VGG()
image=torch.rand(6,3,224,224)
c=model(image)
print(c.shape)


