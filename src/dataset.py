import os
import copy
import math
import random
import pickle
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset   


class ApproxDataset(Dataset):

    def __init__(self, X, Y=None):
        self.data, self.label = X, Y
        
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index] if self.label is not None else None

        return {'data': data, 'label': label}

    def __len__(self):
        return len(self.data)





if __name__ == '__main__':
    pass