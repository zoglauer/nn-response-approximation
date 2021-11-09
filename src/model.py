import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision.models as models
# from src.module.resnet import ResNet18
# from src.module.transpose_cnn import TransposeCNN


class ApproxModel(nn.Module):
    '''
    Model wrapper
    '''

    def __init__(self, model_type='fc', *args, **kwargs):     
        super().__init__()        
        if model_type == 'fc':
            self.model = fully_connected(*args, **kwargs)
            
        elif model_type == 'resnet':
            self.model = ResNet18(*args, **kwargs)

        elif model_type == 'transpose_conv':
            self.model = TransposeCNN(*args, **kwargs)
        
        elif model_type == 'sphere_conv':
            self.model = TransposeCNN(*args, **kwargs)
        
        else:
            raise NotImplementedError
        
        self.model.__name__ = model_type


    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    


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
        nn.ReLU(),
    )
