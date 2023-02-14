import torch
from torch import nn
from torchvision import models
import numpy as np

class Conv_CNN(nn.Module):
    def __init__(self, input_size=2, num_z_grids=4, num_xy_grids=30):
        super().__init__()
        # Future (still under discussion)
        # input size = (1, 64, 64, 2) with batch size = 1 in the first dimension
        # output size = (1, 64*64, 30, 30, 4) with num_xy_grid = 30 and num_z_grid = 4

        # Current
        # input size = (1024, 2)
        # output size = (1024, 30, 30, 4)
        self.num_z_grids = num_z_grids
        self.num_xy_grids = num_xy_grids
        self.fc = nn.Sequential(
            nn.Linear(2, num_xy_grids*num_xy_grids*100),
            nn.ReLU()
        )

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, num_z_grids*10, kernel_size=5, stride=5), 
            nn.ReLU(),
            nn.Conv2d(num_z_grids*10, num_z_grids*10, kernel_size=2, stride=2),  
            nn.ReLU(),
            nn.Conv2d(num_z_grids*10, num_z_grids, kernel_size=1, stride=1), 

        )
        
    
    def forward(self, x):
        # x: (bs, 2)
        x = self.fc(x) # (bs, 90000)
        x = x.view(-1, 1, self.num_xy_grids*10, self.num_xy_grids*10) # (bs, 1, 300, 300)
        x = self.conv_layers(x) # (bs, 4, 30, 30)
        return x.permute(0, 2, 3, 1)  # (bs, 30, 30, 4)

if __name__ == '__main__':
    net = Conv_CNN()
    x = torch.zeros((3, 2))
    y = net(x)
    print(y.shape)
    from torchsummary import summary
    summary(net.cuda(), (3, 2))