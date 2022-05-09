import torch
from torch import nn
from torchvision import models
import numpy as np

class FC(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.0):
        super().__init__()
        self.input_size = input_size
        # self.num_xy_grids = num_xy_grids
        # self.num_z_grids = num_z_grids
        
        # output_size = num_xy_grids * num_xy_grids * num_z_grids
        self.fc_layer = nn.Sequential(
            nn.Linear(input_size, 10),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(10, 100),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            nn.Linear(100, 1000),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
            #nn.Linear(1000, 10000),
            #nn.Dropout(dropout_rate),
            #nn.ReLU(),
            nn.Linear(1000, output_size),
            nn.Dropout(dropout_rate),
            nn.ReLU(),
        )
        self.fc_layer.apply(self.init_weights)
        
    
    def forward(self, x):
        return self.fc_layer(x)
        # return self(x).view(-1, self.num_xy_grids, self.num_xy_grids, self.num_z_grids)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # print("init by kaiming_normal_")
            torch.nn.init.kaiming_normal_(m.weight)
            # kaiming_uniform_
            m.bias.data.fill_(0.00)
    