import torch
import cv2
import numpy as np

from models.UNet import UNet



def load_model(model_weights_path, in_ch, num_cl, base_ch, device):
    model = UNet(in_ch=in_ch, num_cl=num_cl, base_ch=base_ch).to(device)
    model.load_state_dict( torch.load(model_weights_path, map_location=device) )
    model.eval()

    return model


def preprocess_image(image_path, transforms):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Image not found: {image_path}")

    orig_h, orig_w = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms(image=img)['image']

    return img, (orig_h, orig_w)
    

def postprocess_pred(pred, orig_size):
    orig_h, orig_w = orig_size

    mask = pred[0, 0].detach().cpu().numpy()
    mask = (mask * 255).clip(0, 255).astype(np.uint8)
    
    
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return mask


def predict_single_image(img_path, model, transforms, device):
    img, orig_size = preprocess_image(img_path, transforms)
    
    x = img.unsqueeze(dim=0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))
    
    mask = postprocess_pred(pred, orig_size)

    return mask