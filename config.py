IMG_CHANNELS = 3
DOWN_CHANNELS = [32, 48, 64]
NUM_RES = 5
UP_CHANNELS = [48, 32]
FINAL_CHANNELS = IMG_CHANNELS

VGG19_CFG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
    "M",
]

VGG16_CFG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    "M",
]

LOSS_NET_CONFIG = {
    "VGG19": {
        "model_cfg": VGG19_CFG,
        "model_map": {
            "ReLU1_2": 3,
            "ReLU2_2": 8,
            "ReLU3_2": 13,
            "ReLU4_2": 22,
        },
        "content_layer": "ReLU4_2",
    },
    "VGG16": {
        "model_cfg": VGG16_CFG,
        "model_map": {
            "ReLU1_2": 3,
            "ReLU2_2": 8,
            "ReLU3_3": 15,
            "ReLU4_3": 22,
        },
        "content_layer": "ReLU2_2",
    },
}
