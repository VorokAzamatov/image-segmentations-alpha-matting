import torch
import cv2
import numpy as np

from PIL import Image

from configs.config import DEVICE
from train import model
from datasets.datasets import get_val_transforms

def predict_single_image(img_path, model, transforms, device):
    with torch.no_grad():
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = transforms(image=img)['image']
        
        x = img.unsqueeze(dim=0).to(device)
    
        pred = torch.sigmoid(model(x))
    
        mask = pred[0, 0].detach().cpu().numpy()
        mask = (mask * 255).clip(0, 255).astype(np.uint8)
        mask = np.array(Image.fromarray(mask).resize((1024, 1024), resample=Image.BILINEAR))

    return mask


img_path = ''
transforms = get_val_transforms

mask = predict_single_image(img_path, model, transforms, DEVICE)