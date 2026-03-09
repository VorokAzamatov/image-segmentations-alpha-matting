import torch
import cv2
import numpy as np

from pathlib import Path

from models.model_factory import get_model



def load_model(model_type, model_weights_path, device, in_ch=3, num_cl=1, base_ch=32):

    model = get_model(model_type, in_ch, num_cl, base_ch, device)

    state_dict = torch.load(model_weights_path, map_location=device)

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model


def preprocess_image(image_input, transforms):

    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)

    elif hasattr(image_input, "read"):
        image_input.seek(0)
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif isinstance(image_input, np.ndarray):
        img = image_input

    else:
        raise ValueError("Unsupported image input type")

    if img is None:
        raise ValueError("Failed to load image")

    orig_h, orig_w = img.shape[:2]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transforms(image=img)["image"]

    return img, (orig_h, orig_w)
    

def postprocess_pred(pred, orig_size):

    orig_h, orig_w = orig_size

    mask = pred[0, 0].detach().cpu().numpy()
    mask = (mask * 255).clip(0, 255).astype(np.uint8)
    
    
    mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    return mask


def predict_single_image(img_input, model, transforms, device):
    
    img, orig_size = preprocess_image(img_input, transforms)
    
    x = img.unsqueeze(dim=0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(x))
    
    mask = postprocess_pred(pred, orig_size)

    return mask