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


class ConvExpand(Module):
    def __init__(self, linear_layers, conv_layers, config):
        super().__init__()

        # Put encoder layers in Sequential container
        # First increase from 1 --> 64 channels
        # Keep decreasing number of channels

        self.linear_layers = linear_layers
        self.conv_layers = conv_layers

        self.mid_rect_size = config["MID_IMAGE_DIM"]

        # Depth of middle image representation
        self.num_channels = config["MID_IMAGE_DEPTH"]

    # Run x through each layer
    def forward(self, x):
        for layer in self.linear_layers:
            x = layer(x)

        # IMPORTANT: Here, the linear layer output is reshape into rectangle form.
        # New dimeinsions: (BATCH SIZE, NUM_CHANNELS, WIDTH, LENGTH)
        # Batch size could be replaced as first parameter if batches are the same size. Usually not though,
        # so default to how many ever elements there are in the batch.
        x = x.view(
            x.size(0), self.num_channels, self.mid_rect_size[0], self.mid_rect_size[1]
        )

        for layer in self.conv_layers:
            x = layer(x)

        return x
