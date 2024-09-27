
import torch 
import matplotlib.pyplot as plt

def sample(size, alpha, std=0.5, weights=[1/3, 1/3, 1/3], device='cpu', delta=0.0):
    """
    Generate a PyTorch matrix filled with samples from a Gaussian Mixture Model.
    
    Parameters:
    - size: The size of the matrix to generate (e.g., (rows, cols)).
    - alpha: The distance between the centers of the Gaussians.
    - std: The standard deviation of each Gaussian (optional, default is 1.0).
    - weights: The mixture weights for the three peaks (optional, default is equal weighting).
    - device: The device to place the tensor on (optional, default is 'cpu').
    
    Returns:
    - A matrix of the specified size with values sampled from the GMM.
    """
    # Define centers for the Gaussian mixture components
    centers = torch.tensor([-alpha, 0, alpha+delta], device=device)
    
    # Sample from a uniform distribution to decide which Gaussian component to sample from
    component_choices = torch.multinomial(torch.tensor(weights, device=device), size[0] * size[1], replacement=True)
    
    # Generate samples from the Gaussian components
    samples = torch.randn(size[0] * size[1], device=device) * std
    
    # Adjust the samples based on the chosen Gaussian component
    samples += centers[component_choices]
    
    # Reshape the result to the desired matrix size
    return samples.view(size)

def test_func(func, range=[-2, 2]):
    values = []
    for t in torch.linspace(*range, 100):
        values.append(func(t))
    plt.plot(torch.linspace(*range, 100), values)
    plt.show()



def Reg1(W):
    # Square of each element
    wi_squared = torch.abs(W) 
    
    # Absolute value of each element
    abs_wi = torch.abs(W)
    
    # (1 - |wi|)^2
    one_minus_abs_wi_squared = torch.abs(1 - abs_wi)
    
    # Element-wise multiplication of wi^2 and (1 - |wi|)^2
    result = wi_squared * one_minus_abs_wi_squared
    
    # Sum over all elements
    return torch.sum(result)


def threshold_penalty_loss(W, threshold=0.5):
    """
    Penalizes weights whose absolute value exceeds the specified threshold.
    Args:
        W (torch.Tensor): The weight tensor.
        threshold (float): The threshold value.
    Returns:
        torch.Tensor: The computed loss.
    """
    return torch.mean(torch.relu(torch.abs(W) - threshold))


def quantize_ternary(W, threshold=0.5):
    """
    Quantizes the weights to ternary levels based on the threshold.
    Args:
        W (torch.Tensor): The weight tensor.
        threshold (float): The threshold value.
    Returns:
        torch.Tensor: The quantized weight tensor.
    """
    Q = W.clone()
    Q[W > threshold] = 1.0
    Q[W < -threshold] = -1.0
    Q[(-threshold <= W) & (W <= threshold)] = 0.0
    return Q

def quantization_difference_loss(W, threshold=0.5):
    """
    Computes the mean squared difference between the weights and their quantized versions.
    Args:
        W (torch.Tensor): The weight tensor.
        threshold (float): The threshold value for quantization.
    Returns:
        torch.Tensor: The computed loss.
    """
    Q = quantize_ternary(W, threshold)
    return torch.mean((W - Q) ** 2)




test_func(quantization_difference_loss)