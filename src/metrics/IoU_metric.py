import torch

def iou_metric(pred, y, threshold=0.5):
    pred = torch.sigmoid(pred) 

    pred_bin = (pred > threshold).float()

    intersection = (pred_bin * y).sum(dim=[1, 2, 3])
    union = pred_bin.sum(dim=[1, 2, 3]) + y.sum(dim=[1, 2, 3]) - intersection 
    
    IoU = intersection / union.clamp(min=1e-6)

    return torch.mean(IoU)