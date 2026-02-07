import torch

def mse_metric(pred, y):
    pred = torch.sigmoid(pred)

    return torch.mean((pred - y)**2)