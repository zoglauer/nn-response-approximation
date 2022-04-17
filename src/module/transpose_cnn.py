import torch
from torch import nn
from torchvision import models
import numpy as np

class TransposeCNN(nn.Module):
    def __init__(self, input_size=2, num_z_grids=4, num_xy_grids=30):
        super().__init__()
        
        self.num_xy_grids = num_xy_grids
        self.fc = nn.Sequential(
            nn.Linear(2, num_xy_grids*num_xy_grids*100),
            nn.ReLU()
        )
        self.trans_conv = nn.Sequential(
            nn.ConvTranspose2d(1, num_z_grids*5, kernel_size=3, stride=3), 
            nn.ReLU(),
            nn.Conv2d(num_z_grids*5, num_z_grids*5, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.Conv2d(num_z_grids*5, num_z_grids, kernel_size=10, stride=10)
        )
    
    def forward(self, x):
        # x.shape = (bs, 2)
        x = self.fc(x) # (bs, 90000)
        x = x.view(-1, 1, self.num_xy_grids*10, self.num_xy_grids*10) # (bs, 1, 300, 300)
        x = self.trans_conv(x) # (bs, 4, 300, 300)
        # print(x.shape)
        return x.permute(0, 2, 3, 1) # (bs, 30, 30, 4)

   
if __name__ == '__main__':
    pass