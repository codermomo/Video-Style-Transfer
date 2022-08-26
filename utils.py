import random

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


# https://thomasdougherty.ai/pytorch-video-style-transfer/
def get_constant_noise():
    noise_frame = np.zeros(
        (config.IMG_CHANNELS, config.IMG_SIZE, config.IMG_SIZE),
        dtype=np.float32,
    )

    for _ in range(config.NOISE_COUNT):
        x, y = (
            random.randrange(config.IMG_SIZE),
            random.randrange(config.IMG_SIZE),
        )
        noise_frame[0][x][y] += random.uniform(
            -config.NOISE_RANGE, config.NOISE_RANGE
        )
        noise_frame[1][x][y] += random.uniform(
            -config.NOISE_RANGE, config.NOISE_RANGE
        )
        noise_frame[2][x][y] += random.uniform(
            -config.NOISE_RANGE, config.NOISE_RANGE
        )

    return noise_frame


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint ...")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint ...")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def load_model(checkpoint_file, model, device):
    print("=> Loading trained model ...")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])


def image2tensor(img, max_size: int, device: str):
    transform = A.Compose(
        [
            A.geometric.resize.LongestMaxSize(max_size=max_size),
            A.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255
            ),  # [-1, 1]
            ToTensorV2(),
        ]
    )
    tensor = transform(image=img)["image"].to(device)
    tensor = tensor.unsqueeze(dim=0)
    return tensor


def cvimage2tensor(img, max_size: int, device: str):
    img[[0, 2]] = img[[2, 0]]
    return image2tensor(img, max_size, device)


def tensor2cvimage(tensor):
    # [1, C, H, W] -> [C, H, W]
    tensor = tensor.squeeze()
    img = tensor.cpu().numpy()
    # RGB -> BGR
    img[[0, 2]] = img[[2, 0]]
    # [C, H, W] -> [H, W, C]
    img = img.transpose(1, 2, 0)
    img = img * (0.5 * 255) + (0.5 * 255)
    img = img.astype(np.uint8)
    return img