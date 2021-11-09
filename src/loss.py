import torch
import torch.nn.functional as F
import math


class ApproxLoss():
    def __init__(self, loss_type='mse_loss'):
        self.loss_type = loss_type

        if hasattr(torch.nn, loss_type):
            self.criterion = getattr(torch.nn, loss_type)()
        else:
            # Add some self-defined metrics here
            raise NotImplementedError
    
    def __call__(self, pred, target):
        return self.criterion(pred, target)