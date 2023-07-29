import torch
import torch.nn as nn


class BernoulliNoiseInjectionLayer(nn.Module):
    def __init__(self):
        super(BernoulliNoiseInjectionLayer, self).__init__()
        self.probability = nn.Parameter(
            torch.tensor(0.03)
        )  # Initialize with 0.5 probability

    def forward(self, x):
        # Create a binary mask with values sampled from a Bernoulli distribution
        mask = torch.bernoulli(self.probability * torch.ones_like(x))
        # Set the masked pixels to 1
        noisy_x = x + mask
        # Clamp the values to 1 (if noisy_x is larger than 1)
        noisy_x = torch.clamp(noisy_x, min=0, max=1)
        return noisy_x
