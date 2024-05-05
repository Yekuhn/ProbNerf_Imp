import torch

def add_gaussian_noise(tensor, mean=0, stddev=1):
    noise = torch.randn_like(tensor) * stddev + mean
    return tensor + noise

def perturb_weights(weights, alpha=0.05):
    return add_gaussian_noise(weights, stddev=alpha)

# Usage
if __name__ == "__main__":
    sample_weights = torch.randn(10, 10)  # Example weights
    perturbed_weights = perturb_weights(sample_weights, alpha=0.1)
    print("Sample Weights:", sample_weights)
    print("Perturbed Weights:", perturbed_weights)
