import sys
import torch

import pickle
import numpy as np
import platform

from torch.nn import (
    Module,
    Conv2d,
    Sequential,
    ConvTranspose2d,
    ReLU,
    MaxPool2d,
    Linear,
    Conv3d,
    Tanh,
    Dropout,
    DataParallel,
    BatchNorm2d,
)

sys.path.append("../model")
from ConvNoiser import ConvNoiser

from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import torch.optim as optim

from torch.nn import MSELoss

sys.path.append("../trainer")
from trainer import Trainer

# Add utils directory to list of directories to search through
sys.path.append("../utils")
from utils import get_device, set_seed, make_dir, save_img, split_dataset

# Add dataset dirrectory
sys.path.append("../dataset")
from NoisyDataset import NoisyDataset

set_seed(2023)


"""

SET CONFIG OBJECT. 

"""
config = {
    "PROJECT": "add-noise",
    # ------------------- #
    "INPUT_DIR": "../../denoised_data/128-cartesian-1024-768",
    "GPU_PARALLEL": False,
    # ------------------- #
    "NSIDE": 128,
    "SHOW_IMAGES": True,
    "RECT": True,
    "NORMALIZE": False,
    # ------------------- #
    "DEPTH": 36,  # 180 / 5
    "train_pct": 0.94,
    "val_pct": 0.05,
    "BATCH_SIZE": 32,
    # ------------------- #
    "EPOCHS": 1000,
    "PATIENCE": 40,
    "LEARNING_RATE": 0.01,
    # ------------------- #
    "LR_PATIENCE": 10,
    "LR_ADAPT_FACTOR": 0.5,
    # ------------------- #
    "base": torch.float32,
    "device": get_device(),
    "system": platform.system(),
    # ------------------- #
    "SAVE_IMAGES": True,
    "IMAGES_SAVE_DIR": "../../logs/noised-images/",
    "IMAGES_SAVE_INTERVAL": 10,
    # ------------------- #
    "DENOISE_THRESHOLD": 50,
    # ------------------- #
}

# Set other attributes that depend on config specifications
config["NUMPIX"] = 12 * config["NSIDE"] ** 2

# IF USING SAVIO, USE THE SCRATCH DIRECTORY
if platform.system() == "Linux":
    config[
        "INPUT_DIR"
    ] = "/global/scratch/users/akotamraju/denoised_data/128-cartesian-1024-768"
    config["IMAGES_SAVE_DIR"] = "/global/scratch/users/akotamraju/noised-images"

# IF USING GPU, DO DATA PARALLELISM
if config["device"] != "cpu":
    config["GPU_PARALLEL"] = True


# Reshapes the 'y' of a datapoint (the healpix array) into the shape of (depth, length, width)
def reshape_data(data):
    return data


dataset = NoisyDataset(config, transform=reshape_data)

train_loader, val_loader, test_loader = split_dataset(
    dataset, config["train_pct"], config["val_pct"], config["BATCH_SIZE"], shuffle=True
)


layers = Sequential(
    # ------------------------- ENCODER ------------------------- #
    # 1/2X CONV BLOCK
    Conv2d(config["DEPTH"], 64, kernel_size=3, stride=2, padding=1),
    BatchNorm2d(64),
    ReLU(),
    # 1/2X CONV BLOCK
    Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
    BatchNorm2d(128),
    ReLU(),
    # 1/2X CONV BLOCK
    Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
    BatchNorm2d(256),
    ReLU(),
    # 1/2X CONV BLOCK
    Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
    BatchNorm2d(512),
    ReLU(),
    # -------------------------         ------------------------- #
    # ------------------------- DECODER ------------------------- #
    # 2X CONV BLOCK
    ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
    Conv2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
    BatchNorm2d(256),
    ReLU(),
    # 2X CONV BLOCK
    ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
    Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
    BatchNorm2d(128),
    ReLU(),
    # 2X CONV BLOCK
    ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
    Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
    BatchNorm2d(64),
    ReLU(),
    # 2X CONV BLOCK
    ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
    Conv2d(
        in_channels=64, out_channels=config["DEPTH"], kernel_size=3, stride=1, padding=1
    ),
    # -------------------------         ------------------------- #
)

model = ConvNoiser(layers, config)

model = model.to(dtype=config["base"], device=config["device"])

# Use data parallelism if specified
if config["GPU_PARALLEL"]:
    model = DataParallel(model)

# Use MSE Loss
# need to specify cpu
criterion = MSELoss().to(dtype=config["base"], device=config["device"])

# Use Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

# Create scheduler to have adaptive learning rate
scheduler = ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=config["LR_ADAPT_FACTOR"],
    patience=config["LR_PATIENCE"],
    verbose=True,
)


trainer = Trainer(
    model, criterion, optimizer, scheduler, config, train_loader, val_loader
)


trainer.train()
