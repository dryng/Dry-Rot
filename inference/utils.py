import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def load_checkpoint(checkpoint, model):
    """[summary]
        Load a saved model 
    Args:
        checkpoint ([type]): checkpoint to load weights from
        model ([type]): model to load weights from
    """
    model.load_state_dict(checkpoint["state_dict"])
    print("=> Model Loaded")


def numpy_to_torch(data):
    """[summary]
        Convert numpy array of image data to torch tensor of the correct shape 
        The correct shape is (C, H, W). Depending on how the image is loaded (opencv, pillow, etc)
        the color channel may be in the 1st dimension.
    Args:
        data (numpy array): numpy array to convert to torch
    """
    transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0  
            ),
            ToTensorV2()
        ]
    )
    data = transform(image=data)["image"]

    return data.float()