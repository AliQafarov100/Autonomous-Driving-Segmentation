from src.architectures import unet
from src.architectures import deeplabv3_plus

def get_model(model_name: str, num_classes: int):

    if model_name.lower() == "unet":
        return unet.UNet(num_classes=num_classes)

    elif model_name.lower() == "deeplabv3":
        return deeplabv3_plus.DeepLabV3(num_classes=num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")


