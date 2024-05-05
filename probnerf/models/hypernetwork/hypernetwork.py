import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperNetwork(nn.Module):
    def __init__(self, use_perturbation=True):
        super(HyperNetwork, self).__init__()
        self.use_perturbation = use_perturbation
        self.layer1 = nn.Linear(128, 512)
        self.layer2 = nn.Linear(512, 512)
        self.output_layer = nn.Linear(512, 20868)  # Adjust output size as needed

    def forward(self, x, alpha=0.1):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        if self.use_perturbation:
            noise = torch.randn_like(x) * alpha
            return x + noise
        else:
            return x

def generate_parameters(z, use_perturbation=True):
    model = HyperNetwork(use_perturbation=use_perturbation)
    model.eval()  # Assuming inference mode by default
    with torch.no_grad():
        params = model(z)
    return params
