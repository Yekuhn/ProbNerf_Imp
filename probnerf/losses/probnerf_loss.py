import torch
import torch.nn as nn
from .pixelwise import PixelLoss 

class ELBOLoss(nn.Module):
    def __init__(self):
        super(ELBOLoss, self).__init__()
        self.pixel_loss_fn = PixelLoss(option='mse')  # Using MSE for reconstruction loss
        self.kl_loss_weight = 1.0  # Weight of the KL divergence term, adjust as needed

    def forward(self, predictions, targets, mu, log_var):
        # Pixelwise reconstruction loss
        recon_loss = self.pixel_loss_fn(predictions, targets)

        # KL divergence
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

        # ELBO is the sum of the negative reconstruction loss and the negative KL divergence
        elbo_loss = recon_loss + self.kl_loss_weight * kld_loss.mean()

        return elbo_loss

