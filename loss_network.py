import torch
import torch.nn as nn

import config
from config import LOSS_NET_CONFIG

cfg = LOSS_NET_CONFIG["model_cfg"]
output_features_modules_map = LOSS_NET_CONFIG["model_map"]
img_channels = config.IMG_CHANNELS


class LossNetwork(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = self.make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.load_state_dict(torch.load(LOSS_NET_CONFIG["path"]))

    def make_layers(self, cfg):
        layers = []
        in_channels = img_channels
        for out_channels in cfg:
            if out_channels == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(
                    nn.Conv2d(
                        in_channels, out_channels, kernel_size=3, padding=1
                    )
                )
                layers.append(nn.ReLU(inplace=True))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        outputs = {}
        for i, layer in enumerate(
            list(self.features.modules())[
                1 : max(  # noqa: E203
                    list(output_features_modules_map.values())
                )
                + 2
            ]
        ):
            x = layer(x)
            if i in output_features_modules_map.values():
                outputs[i] = x
        return outputs


def test():
    img_channels = 3
    img_size = 512
    x = torch.randn((7, img_channels, img_size, img_size))
    gen = LossNetwork(LOSS_NET_CONFIG["path"])
    print([i.shape for i in gen(x).values()])


if __name__ == "__main__":
    test()
