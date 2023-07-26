import os
import pickle
from torch.utils.data import Dataset
import torch
import numpy as np
import sys

sys.path.append("../utils")
from utils import normalize


class NoisyDataset(Dataset):
    """Dataset with noised and denoised versions of data."""

    def __init__(self, config, transform=None):
        # Stores filepaths of all the data
        self.data_paths = []

        input_dir = config["INPUT_DIR"]

        for filename in os.listdir(input_dir):
            # Load file with pickle
            inp_path = os.path.join(input_dir, filename)

            self.data_paths.append(inp_path)

        self.input_dir = input_dir
        self.transform = transform
        self.config = config

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):
        # Open the file at the given index
        f = open(self.data_paths[index], "rb")
        data = pickle.load(f)
        f.close()

        # Transform if needed
        if self.transform:
            data = self.transform(data)

        x = data["x"]
        y = data["y"]

        # Convert to tensors
        x = torch.tensor(x).to(dtype=self.config["base"], device=self.config["device"])
        y = torch.tensor(y).to(dtype=self.config["base"], device=self.config["device"])

        return x, y
