import os
import torch
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

class PytorchUtils:
    def __init__(self, device=None):
        self.set_device(device)

    def set_device(self, device):
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
    def from_numpy(self, *args, **kwargs):
        return torch.from_numpy(*args, **kwargs).float().to(self.device)

    def to_numpy(self, tensor):
        return tensor.to('cpu').detach().numpy()

    def to_device(self, tensor):
        return tensor.float().to(self.device)

    def normalize(self, target, mean=None, std=None):
        assert target is not None, 'No target was given.'
        assert isinstance(target, torch.tensor), 'target must be given as torch tensors.'
        mean = mean if mean else target.mean()
        std = std if std else target.std()
        
        return (target - mean) / std
    


    