import segmentation_models_pytorch as smp

from models.UNet import UNet

def get_model(model_type, in_ch, num_cl, base_ch, device):

    if model_type == "custom_unet":
        model = UNet(in_ch=in_ch, num_cl=num_cl, base_ch=base_ch)

    elif model_type == "smp_unet":
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=in_ch,
            classes=num_cl
        )

    else:
        raise ValueError("Unknown model type")

    return model.to(device)