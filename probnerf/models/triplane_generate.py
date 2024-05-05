import torch
import torch.nn as nn

class TriplaneGenerator(nn.Module):
    def __init__(self, output_dim):
        super(TriplaneGenerator, self).__init__()
        self.output_dim = output_dim
        self.conv1 = None
        self.relu = nn.ReLU()
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = None
        self.conv_final = None

    def forward(self, concatenated_features):
        if self.conv1 is None:
            input_dim = concatenated_features.shape[1]
            self.conv1 = nn.Conv2d(input_dim, 128, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(128, self.output_dim, kernel_size=3, padding=1)
            self.conv_final = nn.Conv2d(self.output_dim, self.output_dim, kernel_size=3, padding=1)

        x = self.conv1(concatenated_features)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.up_sample(x)
        x = self.conv_final(x)
        return x
