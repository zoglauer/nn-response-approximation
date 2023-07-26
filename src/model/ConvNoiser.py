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
)

""" 
Adds noise to given denoised Healpix image. 
"""


class ConvNoiser(Module):
    def __init__(self, layers, config):
        super().__init__()

        self.layers = layers
        self.config = config

    # Run x through each layer
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
