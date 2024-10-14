import torch
from torch import nn


class PlantClassifier(nn.Module):
    def __init__(self, depth=3, width=256, height=256, num_classes=14):
        super(PlantClassifier, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(depth, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32), nn.MaxPool2d(kernel_size=3)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2),
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3200, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.25),
            nn.Linear(1024, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.linear_layers(x)
        return x


height, width, depth = 128, 128, 3
n_classes = 14
model = PlantClassifier(height, width, depth, n_classes)
