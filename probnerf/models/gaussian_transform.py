import torch
import torch.nn as nn

class GaussianTransform(nn.Module):
    """
    Dynamically transforms feature vectors into Gaussian parameters (means and log-variances).
    """
    def __init__(self, output_dim):
        super(GaussianTransform, self).__init__()
        self.output_dim = output_dim
        self.to_mean = None
        self.to_log_var = None

    def forward(self, features):
        # Initialize layers dynamically based on the first pass feature dimensions
        if self.to_mean is None or self.to_log_var is None:
            feature_dim = features.size(-1)  # Dynamically get the feature dimension
            self.to_mean = nn.Linear(feature_dim, self.output_dim).to(features.device)
            self.to_log_var = nn.Linear(feature_dim, self.output_dim).to(features.device)

        # Apply transformations to compute means and log variances
        means = self.to_mean(features)
        log_vars = self.to_log_var(features)
        # Ensure that variances are positive; convert log variances to actual variances
        variances = torch.exp(log_vars)

        return means, variances  # Return both means and variances

