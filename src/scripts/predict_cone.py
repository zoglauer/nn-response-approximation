# %%

import sys
import torch

import pickle
import numpy as np

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
from ConvExpand import ConvExpand

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
from HealpixSimDataset import HealpixSimDataset

set_seed(2023)


"""

SET CONFIG OBJECT. 

"""
config = {
    "INPUT_DIR": "../../data/cross-sec-data",
    "DATA_INPUT_DIM": (2, 1),
    "GPU_PARALLEL": False,
    # ------------------- #
    "NSIDE": 128,
    "SHOW_IMAGES": True,
    "RECT": True,
    "NORMALIZE": False,
    # ------------------- #
    "DEPTH": 36,  # 180 / 5
    "train_pct": 0.7,
    "val_pct": 0.15,
    "BATCH_SIZE": 8,
    # ------------------- #
    "EPOCHS": 1000,
    "PATIENCE": 40,
    "LEARNING_RATE": 0.01,
    # ------------------- #
    "LR_PATIENCE": 15,
    "LR_ADAPT_FACTOR": 0.5,
    # ------------------- #
    "base": torch.float32,
    "device": get_device(),
    # NOTE:  THESE DEFINE THE DIMENSIONS OF THE MIDDLE IMAGE
    "MID_IMAGE_DEPTH": 1,
    "MID_IMAGE_DIM": (6, 2),
    "FINAL_IMAGE_DIM": (768, 256),
    # ------------------- #
    "SAVE_IMAGES": True,
    "IMAGES_SAVE_DIR": "../../logs/saved-images/",
    "IMAGES_SAVE_INTERVAL": 10,
    # ------------------- #
}

# Set other attributes that depend on config specifications
config["NUMPIX"] = 12 * config["NSIDE"] ** 2

# IF USING SAVIO, USE THE SCRATCH DIRECTORY
if config["device"] != "cpu":
    config["INPUT_DIR"] = "/global/scratch/users/akotamraju/data/cross-sec-data"
    config["IMAGES_SAVE_DIR"] = "/global/scratch/users/akotamraju/saved-images"

# IF USING SAVIO, DO DATA PARALLELISM
if config["device"] != "cpu":
    config["GPU_PARALLEL"] = True

# %%


# Reshapes the 'y' of a datapoint (the healpix array) into the shape of (depth, length, width)
def reshape_data(data):
    return {
        "x": data["x"],
        "y": np.reshape(
            data["y"],
            (
                config["DEPTH"],
                config["FINAL_IMAGE_DIM"][0],
                config["FINAL_IMAGE_DIM"][1],
            ),
        ),
    }


# %%

dataset = HealpixSimDataset(config, transform=reshape_data)

# %%
train_loader, val_loader, test_loader = split_dataset(
    dataset, config["train_pct"], config["val_pct"], config["BATCH_SIZE"], shuffle=True
)

# %%

# IMPORTANT: change linear layer output to batch size * 256 so dimensions match? hmm
lin8 = Sequential(
    Linear(2, 12),
    ReLU(),
    # Linear(12, 48),
    # ReLU(),
    # Linear(48, 192),
    # ReLU(),
    # Linear(384, 3840),
    # ReLU(),
    # Linear(3840, 27648),
    # ReLU(),
)


conv8 = Sequential(
    ConvTranspose2d(1, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
    ReLU(),
    ConvTranspose2d(16, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
    ReLU(),
    ConvTranspose2d(32, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
    ReLU(),
    ConvTranspose2d(64, 128, kernel_size=(4, 4), stride=(4, 4)),
    ReLU(),
    ConvTranspose2d(128, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
    ReLU(),
    ConvTranspose2d(256, 36, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1))
    # # INPUTS: 12 by 4
    # # OUTPUTS: 48 by 16
    # ConvTranspose2d(
    #     in_channels=16,
    #     out_channels=32,
    #     kernel_size=4,
    #     stride=4,
    #     padding=0,
    # ),
    # ReLU(),
    # # INPUTS: 48 by 16
    # # OUTPUTS: 192 by 64
    # ConvTranspose2d(
    #     in_channels=32,
    #     out_channels=64,
    #     kernel_size=4,
    #     stride=4,
    #     padding=0,
    # ),
    # ReLU(),
    # # INPUTS: 192 by 64
    # # OUTPUTS: 768 by 256
    # ConvTranspose2d(
    #     in_channels=64,
    #     out_channels=128,
    #     kernel_size=4,
    #     stride=4,
    #     padding=0,
    # ),
    # Conv2d(
    #     in_channels=128,
    #     out_channels=config["DEPTH"],
    #     kernel_size=3,
    #     stride=1,
    #     padding=1,
    # ),
    # ConvTranspose2d(
    #     in_channels=1,
    #     out_channels=4,
    #     kernel_size=(6, 2),
    #     stride=(2, 4),
    #     padding=(1, 0),
    # ),
    # # INPUTS: 6 by 2
    # # OUTPUTS: 12 by 4
    # ConvTranspose2d(
    #     in_channels=config["MID_IMAGE_DEPTH"],
    #     out_channels=128,
    #     kernel_size=4,
    #     stride=2,
    #     padding=1,
    # ),
    # # INPUTS: 12 by 4
    # # OUTPUTS: 24 by 8
    # ConvTranspose2d(
    #     in_channels=128,
    #     out_channels=128,
    #     kernel_size=4,
    #     stride=2,
    #     padding=1,
    # ),
    # # # INPUTS: 24 by 8
    # # # OUTPUTS: 48 by 16
    # # ConvTranspose2d(
    # #     in_channels=128,
    # #     out_channels=64,
    # #     kernel_size=4,
    # #     stride=2,
    # #     padding=1,
    # # ),
    # # INPUTS: 48 by 16
    # # OUTPUTS: 192 by 64
    # ConvTranspose2d(
    #     in_channels=64,
    #     out_channels=48,
    #     kernel_size=4,
    #     stride=4,
    #     padding=0,
    # ),
    # # INPUTS: 192 by 64
    # # OUTPUTS: 384 by 128
    # ConvTranspose2d(
    #     in_channels=64,
    #     out_channels=32,
    #     kernel_size=4,
    #     stride=2,
    #     padding=1,
    # ),
    # ConvTranspose2d(
    #     in_channels=32,
    #     out_channels=config["DEPTH"],
    #     kernel_size=4,
    #     stride=2,
    #     padding=1,
    # ),
    # # INPUTS: 6 by 2
    # # OUTPUTS: 12 by 4
    # ConvTranspose2d(
    #     in_channels=config["MID_IMAGE_DEPTH"],
    #     out_channels=16,
    #     kernel_size=4,
    #     stride=2,
    #     padding=1,
    # ),
    # ReLU(),
    # # INPUTS: 12 by 4
    # # OUTPUTS: 48 by 16
    # ConvTranspose2d(
    #     in_channels=16,
    #     out_channels=16,
    #     kernel_size=4,
    #     stride=4,
    #     padding=0,
    # ),
    # ReLU(),
    # # INPUTS: 48 by 16
    # # OUTPUTS: 192 by 64
    # ConvTranspose2d(
    #     in_channels=16,
    #     out_channels=16,
    #     kernel_size=4,
    #     stride=4,
    #     padding=0,
    # ),
    # ReLU(),
    # # INPUTS: 192 by 64
    # # OUTPUTS: 768 by 256
    # ConvTranspose2d(
    #     in_channels=16,
    #     out_channels=config["DEPTH"],
    #     kernel_size=4,
    #     stride=4,
    #     padding=0,
    # ),
    # ReLU(),
    # Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
    # ReLU(),
    # Conv2d(
    #     in_channels=128,
    #     out_channels=config["DEPTH"],
    #     kernel_size=3,
    #     stride=1,
    #     padding=1,
    # ),
    # ReLU(),
    # # # INPUTS: 768 by 256
    # # # OUTPUTS:
    # # Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=4, padding=1),
    # # ReLU(),
    # # Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
    # # ReLU(),
    # # Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
    # # ReLU(),
    # # INPUTS: 192 by 64
    # # OUTPUTS: 768 by 256
    # ConvTranspose2d(
    #     in_channels=512,
    #     out_channels=config["DEPTH"],
    #     kernel_size=4,
    #     stride=4,
    #     padding=0,
    # ),
)

expand = ConvExpand(lin8, conv8, config)

# %%
model = expand

model = model.to(dtype=config["base"], device=config["device"])

# Use data parallelism if specified
if config["GPU_PARALLEL"]:
    model = DataParallel(model)

# %%

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


scheduler = CyclicLR(
    optimizer,
    base_lr=0.0001,  # Initial learning rate which is the lower boundary in the cycle for each parameter group
    max_lr=config[
        "LEARNING_RATE"
    ],  # Upper learning rate boundaries in the cycle for each parameter group
    step_size_up=1024,  # Number of training iterations in the increasing half of a cycle
    mode="triangular",
    cycle_momentum=False,
)

# NOTE DOWN THE OPTIMIZER & SCHEDULER INTO THE CONFIG


# optimizer = torch.optim.SGD(model.parameters(), lr=config["LEARNING_RATE"])
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer,
#     T_0=20,
#     T_mult=1,
#     eta_min=0.001,
# )


trainer = Trainer(
    model, criterion, optimizer, scheduler, config, train_loader, val_loader
)

# %%

trainer.train()
