import torch
import torch.nn as nn

def sinusoidal_positional_encoding(x, order=10):
    freqs = torch.pow(2.0, torch.arange(order).float())
    angles = x[..., None] * freqs * (torch.pi * 2.0)
    encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return encoding.flatten(start_dim=-2)

class NeRFDensity(nn.Module):
    def __init__(self, input_channels=63, hidden_units=64):
        super(NeRFDensity, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)
    
    def forward(self, x):
        x = sinusoidal_positional_encoding(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.nn.functional.softplus(self.fc3(x))

class NeRFColor(nn.Module):
    def __init__(self, input_channels=127, hidden_units=64):
        super(NeRFColor, self).__init__()
        self.fc1 = nn.Linear(input_channels, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 3)
    
    def forward(self, positions, density, directions):
        positions = sinusoidal_positional_encoding(positions)
        directions = sinusoidal_positional_encoding(directions)
        x = torch.cat([positions, density, directions], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))
