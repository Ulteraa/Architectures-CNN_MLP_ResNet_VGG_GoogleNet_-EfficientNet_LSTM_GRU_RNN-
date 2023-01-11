import torch
import torch.nn as nn
import math
#base model dimension: #expansion, output_channel, repeat, stride, kernel_size
base_model=[[1,16,1,1,3],
            [6,24,2,2,3],
            [6,40,2,2,5],
            [6,80,3,2,3],
            [6,112,3,1,5],
            [6,192,4,2,5],
            [6,320,1,1,3]]
#phi,resolution,drop_rate
phi_value={'b0':(0,224,0.2),
           'b1':(0.5,240,0.2),
           'b2':(1,260,0.3),
           'b3':(2,300,0.3),
           'b4':(3,380,0.4),
           'b5':(4,456,0.4),
           'b6':(5,528,0.5),
           'b7':(6,600,0.5)}
class conv_block(nn.Module):
    def __init__(self,in_chenel,out_chanel,kernel_size,stride,padding,group=1):
        super(conv_block, self).__init__()
        self.conv=nn.Sequential(nn.Conv2d(in_channels=in_chenel,out_channels=out_chanel,kernel_size=kernel_size
                                          ,stride=stride,padding=padding,groups=group),
                                nn.BatchNorm2d(out_chanel),
                                nn.SiLU())
    def forward(self,x):
        return self.conv(x)
class squeeze_excitation(nn.Module):
    def __init__(self,in_chanel,dim_reduction):
        super(squeeze_excitation, self).__init__()
        self.excite=nn.Sequential(nn.AvgPool2d(1),
                                  nn.Conv2d(in_channels=in_chanel,out_channels=dim_reduction,kernel_size=1),
                                  nn.SiLU(),
                                  nn.Conv2d(in_channels=dim_reduction,out_channels=in_chanel,kernel_size=1),
                                  nn.Sigmoid())
    def forward(self,x):
        return x*self.excite(x)
class inverted_resedual_block(nn.Module):
    def __init__(self,in_chanel,out_chanel,expansion_rate,kernel_size,stride,padding):
        reduction=4
        self.survive_rate=0.8
        super(inverted_resedual_block, self).__init__()
        self.expand=in_chanel!=out_chanel
        self.resedual=in_chanel==out_chanel and stride==1
        self.expand_layer=conv_block(in_chenel=in_chanel,out_chanel=in_chanel*expansion_rate,kernel_size=3,stride=1,padding=1)
        expand_dim=in_chanel*expansion_rate
        reduced=expand_dim//4
        if self.expand:
            expand_dim = in_chanel * expansion_rate
        else:
            expand_dim = in_chanel
        self.conv=nn.Sequential(conv_block(in_chenel=expand_dim,out_chanel=expand_dim,kernel_size=kernel_size,group=expand_dim,stride=stride,padding=padding),
                                squeeze_excitation(expand_dim,reduced),
                                nn.Conv2d(in_channels=expand_dim,out_channels=out_chanel,kernel_size=1),
                                nn.BatchNorm2d(out_chanel)
                                )
    def stochastic_depth(self,x):
        if not self.training:
            return x
        binary=torch.rand(x.shape[0],1,1,1)<self.survive_rate
        return x*binary

    def forward(self,x):
        input_=x
        if self.expand:
            x=self.expand_layer(x)
        if self.resedual:
            return input_+self.stochastic_depth(self.conv(x))
        else:
            return self.conv(x)
class efficient_net(nn.Module):
    def __init__(self,version, class_num):
        super(efficient_net, self).__init__()
        phi,res,drop_=phi_value[version]
        width_factor,depth_factor=self.calculate(phi, alpha=1.2, beta=1.1)
        last_channel=math.ceil(width_factor* 1280)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.feature=self.compute_feature(width_factor,depth_factor,last_channel)
        self.classifier=nn.Sequential(nn.Dropout(drop_),
                                      nn.Linear(last_channel,class_num))
    def calculate(self,phi, alpha=1.2, beta=1.1):
        width_factor=beta**phi; depth_factor=alpha**phi
        return width_factor,depth_factor
    def compute_feature(self,width_factor,depth_factor,last_chanel):
        feature=[]
        conv_=nn.Conv2d(in_channels=3,out_channels=32,stride=2,kernel_size=3,padding=1)
        in_chanel=32
        feature.append(conv_)
        for expansion, output_channel, repeat, stride, kernel_size in base_model:
            out_chanel=int(4*(output_channel*width_factor//4))
            layers_=math.ceil(repeat*depth_factor)

            for layers_num in range(layers_):
                if layers_num==0:
                    stride_ = stride
                else:
                    stride_ = 1
                add_=inverted_resedual_block(in_chanel,out_chanel,expansion,kernel_size,stride_,kernel_size//2)
                feature.append(add_)
                in_chanel=out_chanel
        add_=nn.Conv2d(in_channels=in_chanel, out_channels=last_chanel, kernel_size=1, padding=0, stride=1)
        feature.append(add_)

        return nn.Sequential(*feature)
    def forward(self,x):
        x=self.feature(x)
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.classifier(x)
        return x

def test(version):
    phi,res, drop_=phi_value[version]
    batch=torch.randn(2,3,res,res)
    out_=efficient_net(version, 5)
    x=out_(batch)
    print(x)
test('b7')












