import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision.models as models
# from src.module.resnet import ResNet18
from src.module.transpose_cnn import TransposeCNN
from src.module.conv import Conv_CNN
from src.module.fc import FC
from src.module.resnet import ResNet
from src.module.filter import Filter


class ApproxModel(nn.Module):
    '''
    Model wrapper
    '''

    def __init__(self, model_type='fc', filter_size=3, *args, **kwargs):     
        super().__init__()   
   
        if model_type == 'fc':
            self.model = FC(*args, **kwargs)
            self.model.apply(self.init_weights)
            
        elif model_type == 'resnet':
            self.model = ResNet(*args, **kwargs)

        elif model_type == 'transpose_conv':
            self.model = TransposeCNN(*args, **kwargs)
        
        elif model_type == 'conv':
            self.model = Conv_CNN(*args, **kwargs)
        
        elif model_type == 'sphere_conv':
            pass
            # self.model = TransposeCNN(*args, **kwargs)
        
        else:
            raise NotImplementedError
        
        self.filter_layer = Filter(filter_size)
        self.model_type = model_type
        self.model.__name__ = model_type


    def forward(self, *args, **kwargs):
        x = self.model(*args, **kwargs)  # (bs, 30, 30, 4)
        # bs, w, h, c = x.shape
        # x = self.filter_layer(x.reshape(-1, w, h)).view(bs, w, h, c)
        
        # if self.model_type == 'fc':
        #     x = x.view(bs, -1)
        
        return x
    

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # print("init by kaiming_normal_")
            torch.nn.init.kaiming_normal_(m.weight)
            # kaiming_uniform_
            m.bias.data.fill_(0.00)
    
   


def fully_connected(input_size, output_size, dropout_rate=0.5):
    return nn.Sequential(
        nn.Linear(input_size, 10),
        nn.ReLU(),
        # nn.Dropout(dropout_rate),
        nn.Linear(10, 100),
        nn.ReLU(),
        # nn.Dropout(dropout_rate),
        nn.Linear(100, 1000),
        nn.ReLU(),
        # nn.Dropout(dropout_rate),
        nn.Linear(1000, output_size),
    )


