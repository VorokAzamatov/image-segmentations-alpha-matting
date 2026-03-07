import click
import torch

import os

from PIL import Image

from inference.pipeline import load_model, predict_single_image
from configs.config import DEVICE, IN_CH, BASE_CH, NUM_CL
from data.transforms import get_val_transforms



@click.command()

@click.option("--model_type", "-mt", required=True, help="Model type to use")
@click.option("--img_path", "-i", required=True, help="Path to input image")
@click.option("--weights_path", "-w", required=True, help="Path to model weights")
@click.option("--output_path", "-o", required=True, help="Path to save predicted mask")
@click.option("--img_size", default=512, type=int, help="Resize image to this size for inference")
@click.option("--device", "-d", default=DEVICE, help="Device using for inference")

def main(model_type, img_path, weights_path, output_path, img_size, device):
    device = torch.device(device)

    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Input image not found: {img_path}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    transforms = get_val_transforms(img_size)
    model = load_model(model_type, weights_path, IN_CH, NUM_CL, BASE_CH, device=device)

    mask = predict_single_image(img_path, model, transforms, device=device)


    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    Image.fromarray(mask).save(output_path)

    print(f"Output mask saved to {output_path}")
    


if __name__ == '__main__':
    main()