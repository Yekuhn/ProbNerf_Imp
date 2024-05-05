import torch
from nerf_networks import NeRFDensity, NeRFColor

class ModelProbNeRF(nn.Module):
    def __init__(self):
        super(ModelProbNeRF, self).__init__()
        self.density_net = NeRFDensity()
        self.color_net = NeRFColor()
    
    def forward(self, positions, directions):
        density = self.density_net(positions)
        color = self.color_net(positions, density, directions)
        return density, color
