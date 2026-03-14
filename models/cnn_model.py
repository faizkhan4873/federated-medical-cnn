import torch
import torch.nn as nn

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        # Convolution Layers
        self.conv_layers = nn.Sequential(

            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)

        )

        # Fully Connected Layers
        self.fc_layers = nn.Sequential(

            nn.Linear(128 * 26 * 26, 128),
            nn.ReLU(),

            nn.Linear(128, 2)

        )

    def forward(self, x):

        x = self.conv_layers(x)

        x = x.view(x.size(0), -1)

        x = self.fc_layers(x)

        return x