import torch


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
    data = torch.from_numpy(data)

    # (h, w, c) 
    if data.shape[2] == 3:
        data = data.permute(2, 0, 1)
    
    if data.shape[0] != 3:
        raise ValueError("Error: Incorrect Image dimensions. Should be either (H, W, C) or (C, H, W)")

    return data.float()