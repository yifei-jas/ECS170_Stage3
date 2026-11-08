"""A preactivation Wide ResNet, implemented from scratch for 32x32 images."""
import torch
from torch import nn
from torch.nn import functional as F
from local_code.base_class.method import method


class ResidualBlock(nn.Module):
    def __init__(self,cin,cout,stride=1):
        super().__init__()
        self.bn1=nn.BatchNorm2d(cin)
        self.conv1=nn.Conv2d(cin,cout,3,stride=stride,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(cout)
        self.conv2=nn.Conv2d(cout,cout,3,padding=1,bias=False)
        self.shortcut=nn.Conv2d(cin,cout,1,stride=stride,bias=False) if cin!=cout or stride!=1 else None

    def forward(self,x):
        pre=F.relu(self.bn1(x),inplace=False)
        shortcut=self.shortcut(pre) if self.shortcut is not None else x
        out=self.conv1(pre)
        out=self.conv2(F.relu(self.bn2(out),inplace=True))
        return out+shortcut


class Method_CNN_CIFAR_Residual(method,nn.Module):
    def __init__(self,depth=28,width=4):
        method.__init__(self,'Wide residual CNN','Preactivation residual blocks')
        nn.Module.__init__(self)
        if (depth-4)%6:raise ValueError('Depth must be 6n+4')
        n=(depth-4)//6
        widths=[16,16*width,32*width,64*width]
        self.stem=nn.Conv2d(3,16,3,padding=1,bias=False)
        stages=[]
        previous=16
        for i,current in enumerate(widths[1:]):
            for block in range(n):
                stride=2 if i>0 and block==0 else 1
                stages.append(ResidualBlock(previous,current,stride))
                previous=current
        self.blocks=nn.Sequential(*stages)
        self.bn=nn.BatchNorm2d(widths[-1])
        self.fc=nn.Linear(widths[-1],10)
        for m in self.modules():
            if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')

    def forward(self,x):
        x=self.blocks(self.stem(x))
        x=F.relu(self.bn(x),inplace=True).mean(dim=(2,3))
        return self.fc(x)

    @torch.inference_mode()
    def run(self,x):
        self.eval()
        return self(x).argmax(1)
