import os
from PIL import Image

import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def get_style_image():
    img = np.array(Image.open(config.STYLE_IMG_PATH).convert("RGB"))
    transform = A.Compose(
        [
            A.CenterCrop(min(img.shape[:2]), min(img.shape[:2])),
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255
            ),  # [-1, 1]
            ToTensorV2(),
        ]
    )
    img = transform(image=img)["image"]
    return img


class ContentDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = root
        self.image_size = config.IMG_SIZE
        self.frames = sorted(os.listdir(root))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        # return normal frame at time t
        frame_t = self.frames[idx]
        frame_t_path = os.path.join(self.root, frame_t)
        frame_t = np.array(Image.open(frame_t_path).convert("RGB"))

        transform = A.Compose(
            [
                A.CenterCrop(min(frame_t.shape[:2]), min(frame_t.shape[:2])),
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    max_pixel_value=255,
                ),  # [-1, 1]
                ToTensorV2(),
            ]
        )
        frame_t = transform(image=frame_t)["image"]

        return frame_t
