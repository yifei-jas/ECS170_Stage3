"""Configurable convolution blocks shared by three dataset-specific classes."""
import torch
from torch import nn
from local_code.base_class.method import method


class Method_CNN(method, nn.Module):
    def __init__(self, channels, input_channels, classes, pool_grid,
                 hidden_dim=128, convolutions_per_block=1, kernel_size=3, dropout=.2, pool_kernel=1):
        method.__init__(self, 'CNN', 'Convolution, batch normalization, ReLU and pooling')
        nn.Module.__init__(self)
        blocks=[]
        previous=input_channels
        for width in channels:
            for _ in range(convolutions_per_block):
                blocks.extend([nn.Conv2d(previous,width,kernel_size,padding=kernel_size//2,bias=False),
                               nn.BatchNorm2d(width), nn.ReLU(inplace=True)])
                previous=width
            blocks.append(nn.MaxPool2d(2))
        self.features=nn.Sequential(*blocks)
        self.pool=nn.AvgPool2d(pool_kernel) if pool_kernel>1 else nn.Identity()
        self.classifier=nn.Sequential(nn.Flatten(),
            nn.Linear(channels[-1]*pool_grid[0]*pool_grid[1],hidden_dim),nn.ReLU(inplace=True),
            nn.Dropout(dropout),nn.Linear(hidden_dim,classes))

    def forward(self,x):
        return self.classifier(self.pool(self.features(x)))

    @torch.inference_mode()
    def run(self, x, batch_size=128):
        self.eval()
        return torch.cat([self(batch).argmax(1) for batch in x.split(batch_size)]).cpu().numpy()
