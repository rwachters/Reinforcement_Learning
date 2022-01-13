from pytorch_device import pytorch_device
import torch
import numpy as np


def convert_to_tensor(x) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(pytorch_device)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        return torch.tensor(x).float().to(pytorch_device)


def convert_to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)
