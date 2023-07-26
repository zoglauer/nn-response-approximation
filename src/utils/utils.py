from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

import os
import torch
import random
import numpy as np


def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


# Splits dataset into train, validation, and test datasets and returns dataloader for each dataset
def split_dataset(dataset, train_pct, val_pct, batch_size, shuffle=True):
    train_size = int(train_pct * len(dataset))
    val_size = int(val_pct * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader


# Scales array between 0 and 1
def scale(arr):
    return (arr - np.min(arr)) / np.max(arr)


def normalize(arr):
    # # Normalize each cross section
    range_val = np.max(arr) - np.min(arr)

    # Ensure that there are non-0 values
    if range_val == 0:
        return arr

    # OLD NORMALIZTION:
    # arr = (arr - np.min(arr)) / range_val

    arr = (arr - np.mean(arr)) / np.std(arr)

    return arr


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def make_dir():
    image_dir = "Saved_Images"
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)


def save_img(img, name):
    img = img.view(img.size(0), 1, 12 * NSIDE // 8, 64 * NSIDE // 8)
    save_image(img, name)
