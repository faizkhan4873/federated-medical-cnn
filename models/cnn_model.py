import torch
import torch.nn as nn


class CNNModel(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5):
        super(CNNModel, self).__init__()

        # Convolution layers
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # -> (B, 32, 112, 112)

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # -> (B, 64, 56, 56)

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # -> (B, 128, 28, 28)

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),        # -> (B, 256, 14, 14)
        )

        # AdaptiveAvgPool removes the hardcoded fc input size —
        # output is always (B, 256, 4, 4) regardless of input resolution
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
