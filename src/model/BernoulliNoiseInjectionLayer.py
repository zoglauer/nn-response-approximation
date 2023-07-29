import torch

import torch.nn as nn


class BernoulliNoiseInjectionLayer(nn.Module):
    def __init__(self):
        super(BernoulliNoiseInjectionLayer, self).__init__()
        self.probability = nn.Parameter(
            torch.tensor(0.5)
        )  # Initialize with 0.5 probability

    def forward(self, x):
        # Create a mask of random binary values with the same shape as the input x
        mask = torch.bernoulli(torch.full_like(x, self.probability))
        # Set the pixels to 1 where the mask is 1
        noisy_x = x + mask
        # Clamp the values to 1 (if noisy_x is larger than 1)
        noisy_x = torch.clamp(noisy_x, min=0, max=1)
        return noisy_x
