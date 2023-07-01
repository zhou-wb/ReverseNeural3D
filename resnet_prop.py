import torch
from torch import nn

import torch
import torch.nn as nn
from torch.nn import functional as F

from torchsummary import summary


class residual_block(nn.Module):
    def __init__(self,input_channel, output_channel, stride=1, downsample=None) -> None:
        super().__init__()
        self. downsample=downsample
        self.conv1=nn.Conv2d(input_channel,output_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(output_channel)
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(output_channel,output_channel,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(output_channel)
        self.relu2=nn.ReLU()
    
    
    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu1(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = out.clone() + identity
        return out
    


class ResNet_Prop(nn.Module):
    def __init__(self,input_channel=3,block_num=30) -> None:
        super().__init__()
        self.input_channel=input_channel
        self.first_layer=nn.Sequential(
            nn.Conv2d(self.input_channel,24,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU()
        )
        self.layer1=self.make_layer(3, 24,block_num=1)
        self.layer2=self.make_layer(24,24,block_num=15)
        self.last_layer=nn.Sequential(
            nn.Conv2d(self.input_channel+24,2,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
    
    def forward(self,x):
        identity = x
        x = self.first_layer(x)
        # out=self.layer1(x)
        out=self.layer2(x)
        out = torch.cat((identity,out),dim=1) # concat channel
        out=self.last_layer(out)
        return out
    
    def make_layer(self, input_channel, output_channel, block_num=30,stride=1):
        layers=[]
        layers.append(residual_block(input_channel,output_channel))
        for _ in (1,block_num):
            layers.append(residual_block(output_channel,output_channel))
        return nn. Sequential(*layers)
            

if __name__ == '__main__':   
    model=ResNet_Prop()
    model= model.cuda()
    summary(model, (3,480,640))
    print(model)