import random

import numpy as np
import torch

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
